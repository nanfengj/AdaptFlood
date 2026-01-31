// lisflood_gpu_core.cpp - 简化的 GPU 核心计算模块
// 用于点源扩散场景，只实现最基本的 Manning 公式流量计算和水深更新

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cmath>
#include <algorithm>

// 类型定义
using RealView = Kokkos::View<double*, Kokkos::CudaUVMSpace>;
using IntView = Kokkos::View<int*, Kokkos::CudaUVMSpace>;

// 辅助函数
KOKKOS_INLINE_FUNCTION double getmax(double a, double b) { return (a > b) ? a : b; }
KOKKOS_INLINE_FUNCTION double getmin(double a, double b) { return (a < b) ? a : b; }

//==============================================================================
// FloodplainQ_GPU - GPU版本的洪泛平原流量计算
// 简化版：只实现基本 Manning 公式，不支持 weir、porosity、acceleration 等
//==============================================================================
void FloodplainQ_GPU(
    RealView& H,        // 水深数组 [xsz * ysz]
    RealView& DEM,      // 高程数组 [xsz * ysz]
    RealView& Qx,       // x方向流量 [(xsz+1) * ysz]
    RealView& Qy,       // y方向流量 [xsz * (ysz+1)]
    RealView& Manningsn, // Manning系数数组（可为空）
    int xsz, int ysz,   // 网格尺寸
    double dx, double dy, // 网格间距
    double FPn,         // 默认Manning系数
    double DepthThresh, // 水深阈值
    double MaxHflow,    // 最大流动水深
    double dhlin,       // 线性化阈值
    double& Tstep,      // 时间步（输入输出）
    double Qlimfact,    // 流量限制系数
    double dA,          // 单元面积
    bool adaptive_ts    // 是否自适应时间步
)
{
    double TmpTstep = Tstep;
    double dx_sqrt = sqrt(dx);

    // 为最小时间步创建 atomic view
    RealView min_timestep("min_timestep", 1);
    Kokkos::deep_copy(min_timestep, TmpTstep);

    bool has_manningsn = (Manningsn.extent(0) > 0);

    //==========================================================================
    // 1. 计算 Qx (x方向流量)
    //==========================================================================
    Kokkos::parallel_for("Calculate_Qx",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {xsz-1, ysz}),
        KOKKOS_LAMBDA(const int i, const int j) {
            const int p0 = i + j * xsz;
            const int p1 = i + 1 + j * xsz;
            const int qx_idx = (i+1) + j * (xsz+1);

            double z0 = DEM(p0);
            double z1 = DEM(p1);
            double h0 = H(p0);
            double h1 = H(p1);

            // 计算 Manning 系数
            double fn = FPn;
            if(has_manningsn) {
                fn = 0.5 * (Manningsn(p0) + Manningsn(p1));
            }

            double Q = 0.0;
            double local_ts = TmpTstep;

            if(h0 > DepthThresh || h1 > DepthThresh) {
                // 从 0 流向 1
                if(z0 + h0 > z1 + h1 && h0 > DepthThresh) {
                    double dh = z0 + h0 - z1 - h1;
                    double Sf = sqrt(dh / dx);
                    double hflow = getmax(z0 + h0, z1 + h1) - getmax(z0, z1);
                    hflow = getmax(hflow, 0.0);
                    hflow = getmin(hflow, MaxHflow);

                    if(hflow > DepthThresh) {
                        double alpha;
                        if(dh < dhlin) {
                            Sf = sqrt(dx / dhlin) * (dh / dx);
                            alpha = (pow(hflow, 5.0/3.0) * dx_sqrt) / (fn * sqrt(dhlin));
                        } else {
                            alpha = pow(hflow, 5.0/3.0) / (2.0 * fn * Sf);
                        }

                        Q = pow(hflow, 5.0/3.0) * Sf * dy / fn;

                        if(adaptive_ts) {
                            local_ts = getmin(local_ts, 0.25 * dy * dy / alpha);
                        } else {
                            double Qlim = Qlimfact * dA * fabs(dh) / (8.0 * TmpTstep);
                            if(fabs(Q) > Qlim) Q = (Q > 0) ? Qlim : -Qlim;
                        }
                    }
                }
                // 从 1 流向 0
                else if(z0 + h0 < z1 + h1 && h1 > DepthThresh) {
                    double dh = z1 + h1 - z0 - h0;
                    double Sf = sqrt(dh / dx);
                    double hflow = getmax(z0 + h0, z1 + h1) - getmax(z0, z1);
                    hflow = getmax(hflow, 0.0);
                    hflow = getmin(hflow, MaxHflow);

                    if(hflow > DepthThresh) {
                        double alpha;
                        if(dh < dhlin) {
                            Sf = sqrt(dx / dhlin) * (dh / dx);
                            alpha = (pow(hflow, 5.0/3.0) * dx_sqrt) / (fn * sqrt(dhlin));
                        } else {
                            alpha = pow(hflow, 5.0/3.0) / (2.0 * fn * Sf);
                        }

                        Q = -pow(hflow, 5.0/3.0) * Sf * dy / fn;

                        if(adaptive_ts) {
                            local_ts = getmin(local_ts, 0.25 * dy * dy / alpha);
                        } else {
                            double Qlim = Qlimfact * dA * fabs(dh) / (8.0 * TmpTstep);
                            if(fabs(Q) > Qlim) Q = (Q > 0) ? Qlim : -Qlim;
                        }
                    }
                }
            }

            Qx(qx_idx) = Q;

            // 更新全局最小时间步
            if(adaptive_ts) {
                Kokkos::atomic_min(&min_timestep(0), local_ts);
            }
        });

    //==========================================================================
    // 2. 计算 Qy (y方向流量)
    //==========================================================================
    Kokkos::parallel_for("Calculate_Qy",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {xsz, ysz-1}),
        KOKKOS_LAMBDA(const int i, const int j) {
            const int p0 = i + j * xsz;
            const int p1 = i + (j+1) * xsz;
            const int qy_idx = i + (j+1) * (xsz+1);

            double z0 = DEM(p0);
            double z1 = DEM(p1);
            double h0 = H(p0);
            double h1 = H(p1);

            // 计算 Manning 系数
            double fn = FPn;
            if(has_manningsn) {
                fn = 0.5 * (Manningsn(p0) + Manningsn(p1));
            }

            double Q = 0.0;
            double local_ts = TmpTstep;

            if(h0 > DepthThresh || h1 > DepthThresh) {
                // 从 0 流向 1
                if(z0 + h0 > z1 + h1 && h0 > DepthThresh) {
                    double dh = z0 + h0 - z1 - h1;
                    double Sf = sqrt(dh / dy);
                    double hflow = getmax(z0 + h0, z1 + h1) - getmax(z0, z1);
                    hflow = getmax(hflow, 0.0);
                    hflow = getmin(hflow, MaxHflow);

                    if(hflow > DepthThresh) {
                        double alpha;
                        if(dh < dhlin) {
                            Sf = sqrt(dy / dhlin) * (dh / dy);
                            alpha = (pow(hflow, 5.0/3.0) * dx_sqrt) / (fn * sqrt(dhlin));
                        } else {
                            alpha = pow(hflow, 5.0/3.0) / (2.0 * fn * Sf);
                        }

                        Q = pow(hflow, 5.0/3.0) * Sf * dx / fn;

                        if(adaptive_ts) {
                            local_ts = getmin(local_ts, 0.25 * dx * dx / alpha);
                        } else {
                            double Qlim = Qlimfact * dA * fabs(dh) / (8.0 * TmpTstep);
                            if(fabs(Q) > Qlim) Q = (Q > 0) ? Qlim : -Qlim;
                        }
                    }
                }
                // 从 1 流向 0
                else if(z0 + h0 < z1 + h1 && h1 > DepthThresh) {
                    double dh = z1 + h1 - z0 - h0;
                    double Sf = sqrt(dh / dy);
                    double hflow = getmax(z0 + h0, z1 + h1) - getmax(z0, z1);
                    hflow = getmax(hflow, 0.0);
                    hflow = getmin(hflow, MaxHflow);

                    if(hflow > DepthThresh) {
                        double alpha;
                        if(dh < dhlin) {
                            Sf = sqrt(dy / dhlin) * (dh / dy);
                            alpha = (pow(hflow, 5.0/3.0) * dx_sqrt) / (fn * sqrt(dhlin));
                        } else {
                            alpha = pow(hflow, 5.0/3.0) / (2.0 * fn * Sf);
                        }

                        Q = -pow(hflow, 5.0/3.0) * Sf * dx / fn;

                        if(adaptive_ts) {
                            local_ts = getmin(local_ts, 0.25 * dx * dx / alpha);
                        } else {
                            double Qlim = Qlimfact * dA * fabs(dh) / (8.0 * TmpTstep);
                            if(fabs(Q) > Qlim) Q = (Q > 0) ? Qlim : -Qlim;
                        }
                    }
                }
            }

            Qy(qy_idx) = Q;

            // 更新全局最小时间步
            if(adaptive_ts) {
                Kokkos::atomic_min(&min_timestep(0), local_ts);
            }
        });

    // 更新时间步
    Kokkos::fence();
    auto host_ts = Kokkos::create_mirror_view(min_timestep);
    Kokkos::deep_copy(host_ts, min_timestep);
    Tstep = host_ts(0);
}

//==============================================================================
// UpdateH_GPU - GPU版本的水深更新
//==============================================================================
void UpdateH_GPU(
    RealView& H,        // 水深数组 [xsz * ysz]
    RealView& Qx,       // x方向流量 [(xsz+1) * ysz]
    RealView& Qy,       // y方向流量 [xsz * (ysz+1)]
    IntView& ChanMask,  // 河道掩码（-1表示非河道）
    int xsz, int ysz,   // 网格尺寸
    double dA,          // 单元面积
    double Tstep        // 时间步
)
{
    Kokkos::parallel_for("UpdateH",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {xsz, ysz}),
        KOKKOS_LAMBDA(const int i, const int j) {
            const int idx = i + j * xsz;

            // 只更新非河道单元
            if(ChanMask(idx) == -1) {
                // 获取进出流量
                double Qx_in  = Qx(i + j * (xsz+1));
                double Qx_out = Qx((i+1) + j * (xsz+1));
                double Qy_in  = Qy(i + j * (xsz+1));
                double Qy_out = Qy(i + (j+1) * (xsz+1));

                // 计算体积变化
                double dV = Tstep * (Qx_in - Qx_out + Qy_in - Qy_out);

                // 更新水深
                H(idx) += dV / dA;

                // 确保非负
                if(H(idx) < 0.0) H(idx) = 0.0;
            }
        });

    Kokkos::fence();
}

//==============================================================================
// AddPointSource_GPU - GPU版本的点源添加
// 支持 QVAR 类型的时变流量点源
//==============================================================================
void AddPointSource_GPU(
    RealView& H,        // 水深数组
    int x, int y,       // 点源位置
    int xsz,            // 网格x方向尺寸
    double Q,           // 流量 (m^3/s)
    double dx,          // 网格间距
    double Tstep,       // 时间步
    double dA           // 单元面积
)
{
    // 计算点源对水深的贡献
    // Q * dx * Tstep / dA 是体积变化量换算成水深
    int idx = x + y * xsz;
    double dH = Q * dx * Tstep / dA;

    // 使用 atomic_add 更新水深（支持多点源）
    Kokkos::parallel_for("AddPointSource", 1, KOKKOS_LAMBDA(int) {
        Kokkos::atomic_add(&H(idx), dH);
    });
}
