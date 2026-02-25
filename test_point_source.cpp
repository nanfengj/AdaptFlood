// test_point_source.cpp - 测试点源扩散的 GPU 计算
// 验证 FloodplainQ_GPU 和 UpdateH_GPU 是否正确工作

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cmath>

// 类型定义
using RealView = Kokkos::View<double*, Kokkos::CudaUVMSpace>;
using IntView = Kokkos::View<int*, Kokkos::CudaUVMSpace>;

// 外部函数声明
extern void FloodplainQ_GPU(
    RealView& H, RealView& DEM, RealView& Qx, RealView& Qy, RealView& Manningsn,
    int xsz, int ysz, double dx, double dy, double FPn, double DepthThresh,
    double MaxHflow, double dhlin, double& Tstep, double Qlimfact, double dA, bool adaptive_ts);

extern void UpdateH_GPU(
    RealView& H, RealView& Qx, RealView& Qy, IntView& ChanMask,
    int xsz, int ysz, double dA, double Tstep);

extern void AddPointSource_GPU(
    RealView& H, int x, int y, int xsz, double Q, double dx, double Tstep, double dA);

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        printf("=== LISFLOOD GPU Point Source Test ===\n");
        printf("Execution space: %s\n", typeid(Kokkos::DefaultExecutionSpace).name());

        // 网格参数
        const int xsz = 50;
        const int ysz = 50;
        const double dx = 10.0;   // 10m 网格
        const double dy = 10.0;
        const double dA = dx * dy; // 100 m^2

        // 模拟参数
        const double FPn = 0.03;         // Manning 系数
        const double DepthThresh = 0.001; // 1mm 水深阈值
        const double MaxHflow = 10.0;     // 最大流动水深
        const double dhlin = 0.01;        // 线性化阈值
        const double Qlimfact = 1.0;      // 流量限制系数
        double InitTstep = 1.0;           // 初始时间步 1s

        // 点源参数
        const int ps_x = xsz / 2;  // 网格中心
        const int ps_y = ysz / 2;
        // LISFLOOD 中 Q 的单位是 m²/s（单位宽度流量），Q * dx 才是体积流量 m³/s
        // 如果想要 1 m³/s 的体积流量，需要 ps_Q = 1.0 / dx = 0.1 m²/s
        const double ps_Q = 0.1;   // 0.1 m²/s -> 实际体积流量 = 0.1 * 10 = 1 m³/s

        // 模拟时间
        const double sim_time = 60.0;  // 模拟60秒
        double t = 0.0;

        // 分配数组
        RealView H("H", xsz * ysz);
        RealView DEM("DEM", xsz * ysz);
        RealView Qx("Qx", (xsz+1) * ysz);
        RealView Qy("Qy", xsz * (ysz+1));
        RealView Manningsn("Manningsn", 0);  // 空数组，使用默认 FPn
        IntView ChanMask("ChanMask", xsz * ysz);

        // 初始化 DEM（平坦地形，中心略低以便水能扩散）
        Kokkos::parallel_for("InitDEM", xsz * ysz, KOKKOS_LAMBDA(int idx) {
            int i = idx % xsz;
            int j = idx / xsz;
            // 中心低，四周高（碗状地形）
            double dist = sqrt(pow(i - xsz/2.0, 2) + pow(j - ysz/2.0, 2));
            DEM(idx) = dist * 0.001;  // 0.1% 坡度
        });

        // 初始化 H = 0
        Kokkos::parallel_for("InitH", xsz * ysz, KOKKOS_LAMBDA(int idx) {
            H(idx) = 0.0;
        });

        // 初始化 ChanMask = -1（全部为非河道）
        Kokkos::parallel_for("InitChanMask", xsz * ysz, KOKKOS_LAMBDA(int idx) {
            ChanMask(idx) = -1;
        });

        // 初始化 Qx, Qy = 0
        Kokkos::parallel_for("InitQx", (xsz+1) * ysz, KOKKOS_LAMBDA(int idx) {
            Qx(idx) = 0.0;
        });
        Kokkos::parallel_for("InitQy", xsz * (ysz+1), KOKKOS_LAMBDA(int idx) {
            Qy(idx) = 0.0;
        });

        Kokkos::fence();

        printf("\nGrid: %d x %d, dx=%.1fm, dy=%.1fm\n", xsz, ysz, dx, dy);
        printf("Point source at (%d, %d), Q=%.2f m3/s\n", ps_x, ps_y, ps_Q);
        printf("Simulating %.1f seconds...\n\n", sim_time);

        // 初始体积
        double init_vol = 0.0;

        // 时间循环
        int step_count = 0;
        while(t < sim_time) {
            double Tstep = InitTstep;

            // 1. 计算流量
            FloodplainQ_GPU(H, DEM, Qx, Qy, Manningsn,
                           xsz, ysz, dx, dy, FPn, DepthThresh,
                           MaxHflow, dhlin, Tstep, Qlimfact, dA, true);

            // 2. 添加点源
            AddPointSource_GPU(H, ps_x, ps_y, xsz, ps_Q, dx, Tstep, dA);

            // 3. 更新水深
            UpdateH_GPU(H, Qx, Qy, ChanMask, xsz, ysz, dA, Tstep);

            t += Tstep;
            step_count++;

            // 每10步输出一次
            if(step_count % 100 == 0) {
                // 计算总体积
                double total_vol = 0.0;
                Kokkos::parallel_reduce("CalcVol", xsz * ysz,
                    KOKKOS_LAMBDA(int idx, double& vol) {
                        vol += H(idx) * dA;
                    }, total_vol);

                // 获取中心水深
                auto H_host = Kokkos::create_mirror_view(H);
                Kokkos::deep_copy(H_host, H);
                double center_H = H_host(ps_x + ps_y * xsz);

                // 计算淹没面积
                double flood_area = 0.0;
                Kokkos::parallel_reduce("CalcArea", xsz * ysz,
                    KOKKOS_LAMBDA(int idx, double& area) {
                        if(H(idx) > 0.01) area += dA;
                    }, flood_area);

                printf("t=%.1fs: Tstep=%.4f, H_center=%.4fm, Vol=%.2fm3, Area=%.0fm2\n",
                       t, Tstep, center_H, total_vol, flood_area);
            }
        }

        // 最终结果
        printf("\n=== Final Results ===\n");

        // 计算总体积
        double final_vol = 0.0;
        Kokkos::parallel_reduce("FinalVol", xsz * ysz,
            KOKKOS_LAMBDA(int idx, double& vol) {
                vol += H(idx) * dA;
            }, final_vol);

        double expected_vol = ps_Q * dx * sim_time;  // 期望体积 = Q * dx * 时间 = 体积流量 * 时间
        double vol_error = fabs(final_vol - expected_vol) / expected_vol * 100;

        printf("Simulation time: %.1f s\n", sim_time);
        printf("Total steps: %d\n", step_count);
        printf("Expected volume: %.2f m3\n", expected_vol);
        printf("Actual volume: %.2f m3\n", final_vol);
        printf("Volume error: %.2f%%\n", vol_error);

        if(vol_error < 1.0) {
            printf("\n*** MASS CONSERVATION TEST PASSED! ***\n");
        } else {
            printf("\n*** WARNING: Volume error > 1%% ***\n");
        }

        // 计算淹没面积
        double flood_area = 0.0;
        Kokkos::parallel_reduce("FinalArea", xsz * ysz,
            KOKKOS_LAMBDA(int idx, double& area) {
                if(H(idx) > 0.01) area += dA;
            }, flood_area);
        printf("Flooded area: %.0f m2\n", flood_area);

        // 获取最大水深
        double max_H = 0.0;
        Kokkos::parallel_reduce("MaxH", xsz * ysz,
            KOKKOS_LAMBDA(int idx, double& maxh) {
                if(H(idx) > maxh) maxh = H(idx);
            }, Kokkos::Max<double>(max_H));
        printf("Max water depth: %.4f m\n", max_H);
    }
    Kokkos::finalize();
    return 0;
}
