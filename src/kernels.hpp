// kernels.hpp - GPU内核函数声明
#ifndef LISFLOOD_KERNELS_HPP
#define LISFLOOD_KERNELS_HPP

#include "types.hpp"

// 更新Qold数组（加速模式需要）
void UpdateQs_GPU(RealView& Qx, RealView& Qy, RealView& Qxold, RealView& Qyold,
                  int xsz, int ysz, double dx);

// GPU流量计算 - 使用LISFLOOD acceleration模式
void FloodplainQ_GPU(RealView& H, RealView& DEM, RealView& Qx, RealView& Qy,
    RealView& Qxold, RealView& Qyold,
    int xsz, int ysz, double dx, double dy, double FPn, double DepthThresh,
    double MaxHflow, double dhlin, double& Tstep, double Qlimfact, double dA,
    double cfl, double g, double InitTstep, double theta, double nodata);

// GPU边界条件处理
void BCs_GPU(RealView& Qx, RealView& Qy, RealView& H, RealView& DEM,
             int xsz, int ysz, double dx, double g, double dt, double nodata,
             bool west_free, bool east_free, bool north_free, bool south_free);

// GPU点边界条件应用
void ApplyPointBCs_GPU(RealView& H, IntView& bc_idx, IntView& bc_type, RealView& bc_value,
                       int num_bcs, int ncols, double dx, double dA, double Tstep);

// GPU水深更新
void UpdateH_GPU(RealView& H, RealView& Qx, RealView& Qy, int xsz, int ysz, double dA, double Tstep);

// GPU流速计算
void ComputeVelocity_GPU(RealView& Vx_out, RealView& Vy_out,
                         RealView& H, RealView& DEM, RealView& Qx, RealView& Qy,
                         int xsz, int ysz, double dx, double nodata);

// GPU降雨计算
void Rainfall_GPU(RealView& H, RealView& DEM, int xsz, int ysz,
                  double rain_rate, double Tstep, double nodata);

#endif // LISFLOOD_KERNELS_HPP
