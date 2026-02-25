// kernels.cpp - GPU内核函数实现
#include "kernels.hpp"
#include <cmath>

// 更新Qold数组（加速模式需要）
void UpdateQs_GPU(RealView& Qx, RealView& Qy, RealView& Qxold, RealView& Qyold,
                  int xsz, int ysz, double dx) {
    double dxinv = 1.0 / dx;

    Kokkos::parallel_for("UpdateQxold", (xsz+1) * (ysz+1), KOKKOS_LAMBDA(int idx) {
        Qxold(idx) = Qx(idx) * dxinv;
    });

    Kokkos::parallel_for("UpdateQyold", (xsz+1) * (ysz+1), KOKKOS_LAMBDA(int idx) {
        Qyold(idx) = Qy(idx) * dxinv;
    });

    Kokkos::fence();
}

// GPU流量计算 - 使用LISFLOOD acceleration模式（自适应时间步）
void FloodplainQ_GPU(RealView& H, RealView& DEM, RealView& Qx, RealView& Qy,
    RealView& Qxold, RealView& Qyold,
    int xsz, int ysz, double dx, double dy, double FPn, double DepthThresh,
    double MaxHflow, double dhlin, double& Tstep, double Qlimfact, double dA,
    double cfl, double g, double InitTstep, double theta, double nodata) {

    // 1. 使用parallel_reduce找最大水深
    double Hmax = 0.0;
    Kokkos::parallel_reduce("FindHmax", xsz * ysz,
        KOKKOS_LAMBDA(int idx, double& max_h) {
            if(DEM(idx) != nodata && H(idx) > max_h) max_h = H(idx);
        }, Kokkos::Max<double>(Hmax));

    // 2. 根据最大水深计算自适应时间步
    double dt;
    if(Hmax > DepthThresh) {
        dt = cfl * dx / sqrt(g * Hmax);
    } else {
        dt = InitTstep;
    }

    Tstep = dt;

    // 计算Qx
    Kokkos::parallel_for("Qx", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{xsz-1,ysz}),
        KOKKOS_LAMBDA(int i, int j) {
            int p0 = i + j * xsz, p1 = i + 1 + j * xsz;
            int pq0 = (i+1) + j * (xsz+1);

            double z0 = DEM(p0), z1 = DEM(p1);

            if(z0 == nodata || z1 == nodata) {
                Qx(pq0) = 0.0;
                return;
            }

            double h0 = H(p0), h1 = H(p1);
            double fn = FPn;
            double y0 = z0 + h0, y1 = z1 + h1;
            double dh = y0 - y1;
            double Sf = -dh / dx;
            double hflow = getmax(y0, y1) - getmax(z0, z1);
            hflow = getmax(hflow, 0.0);
            hflow = getmin(hflow, MaxHflow);

            double Q = 0.0;

            if(hflow > DepthThresh) {
                double q0 = Qxold(pq0);
                double qup = (i > 0) ? Qxold(pq0-1) : 0.0;
                double qdown = (i < xsz-2) ? Qxold(pq0+1) : 0.0;

                double qy_avg = 0.0;
                int pqy1 = i + j*(xsz+1);
                int pqy2 = (i+1) + j*(xsz+1);
                int pqy3 = i + (j+1)*(xsz+1);
                int pqy4 = (i+1) + (j+1)*(xsz+1);
                qy_avg = (Qyold(pqy1) + Qyold(pqy2) + Qyold(pqy3) + Qyold(pqy4)) / 4.0;
                double qvect = sqrt(q0*q0 + qy_avg*qy_avg);

                double numerator = (theta*q0 + 0.5*(1.0-theta)*(qup+qdown)) - (g*dt*hflow*Sf);
                double denominator = 1.0 + g*dt*hflow*fn*fn*fabs(qvect) / pow(hflow, 10.0/3.0);
                Q = numerator / denominator * dx;

                if(Q * dh < 0.0) {
                    numerator = q0 - (g*dt*hflow*Sf);
                    Q = numerator / denominator * dx;
                }
            }

            Qx(pq0) = Q;
        });

    // 计算Qy
    Kokkos::parallel_for("Qy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{xsz,ysz-1}),
        KOKKOS_LAMBDA(int i, int j) {
            int p0 = i + j * xsz, p1 = i + (j+1) * xsz;
            int pq0 = i + (j+1) * (xsz+1);

            double z0 = DEM(p0), z1 = DEM(p1);

            if(z0 == nodata || z1 == nodata) {
                Qy(pq0) = 0.0;
                return;
            }

            double h0 = H(p0), h1 = H(p1);
            double fn = FPn;
            double y0 = z0 + h0, y1 = z1 + h1;
            double dh = y0 - y1;
            double Sf = -dh / dx;
            double hflow = getmax(y0, y1) - getmax(z0, z1);
            hflow = getmax(hflow, 0.0);
            hflow = getmin(hflow, MaxHflow);

            double Q = 0.0;

            if(hflow > DepthThresh) {
                double q0 = Qyold(pq0);
                double qup = (j > 0) ? Qyold(i + j*(xsz+1)) : 0.0;
                double qdown = (j < ysz-2) ? Qyold(i + (j+2)*(xsz+1)) : 0.0;

                double qx_avg = 0.0;
                int pqx1 = i + j*(xsz+1);
                int pqx2 = (i+1) + j*(xsz+1);
                int pqx3 = i + (j+1)*(xsz+1);
                int pqx4 = (i+1) + (j+1)*(xsz+1);
                qx_avg = (Qxold(pqx1) + Qxold(pqx2) + Qxold(pqx3) + Qxold(pqx4)) / 4.0;
                double qvect = sqrt(q0*q0 + qx_avg*qx_avg);

                double numerator = (theta*q0 + 0.5*(1.0-theta)*(qup+qdown)) - (g*dt*hflow*Sf);
                double denominator = 1.0 + g*dt*hflow*fn*fn*fabs(qvect) / pow(hflow, 10.0/3.0);
                Q = numerator / denominator * dx;

                if(Q * dh < 0.0) {
                    numerator = q0 - (g*dt*hflow*Sf);
                    Q = numerator / denominator * dx;
                }
            }

            Qy(pq0) = Q;
        });

    Kokkos::fence();
}

// GPU边界条件处理
void BCs_GPU(RealView& Qx, RealView& Qy, RealView& H, RealView& DEM,
             int xsz, int ysz, double dx, double g, double dt, double nodata,
             bool west_free, bool east_free, bool north_free, bool south_free) {

    // 西边界 (i=0)
    if(west_free) {
        Kokkos::parallel_for("BCs_West_FREE", ysz, KOKKOS_LAMBDA(int j) {
            int p0 = 0 + j * xsz;
            double h0 = H(p0);
            double z0 = DEM(p0);
            if(z0 == nodata || h0 < 0.001) {
                Qx(j * (xsz+1)) = 0.0;
            } else {
                int p1 = 1 + j * xsz;
                double h1 = H(p1);
                double z1 = DEM(p1);
                double y0 = z0 + h0;
                double y1 = z1 + h1;
                if(y0 > z0) {
                    double Sf = (y1 - y0) / dx;
                    if(Sf < 0) {
                        double Q = -h0 * sqrt(g * h0) * dx * 0.5;
                        Qx(j * (xsz+1)) = Q;
                    } else {
                        Qx(j * (xsz+1)) = 0.0;
                    }
                } else {
                    Qx(j * (xsz+1)) = 0.0;
                }
            }
        });
    } else {
        Kokkos::parallel_for("BCs_West_CLOSED", ysz + 1, KOKKOS_LAMBDA(int j) {
            Qx(j * (xsz+1)) = 0.0;
        });
    }

    // 东边界 (i=xsz)
    if(east_free) {
        Kokkos::parallel_for("BCs_East_FREE", ysz, KOKKOS_LAMBDA(int j) {
            int p0 = (xsz-1) + j * xsz;
            double h0 = H(p0);
            double z0 = DEM(p0);
            if(z0 == nodata || h0 < 0.001) {
                Qx(xsz + j * (xsz+1)) = 0.0;
            } else {
                if(h0 > 0.001) {
                    double Q = h0 * sqrt(g * h0) * dx * 0.5;
                    Qx(xsz + j * (xsz+1)) = Q;
                } else {
                    Qx(xsz + j * (xsz+1)) = 0.0;
                }
            }
        });
    } else {
        Kokkos::parallel_for("BCs_East_CLOSED", ysz + 1, KOKKOS_LAMBDA(int j) {
            Qx(xsz + j * (xsz+1)) = 0.0;
        });
    }

    // 北边界 (j=0)
    if(north_free) {
        Kokkos::parallel_for("BCs_North_FREE", xsz, KOKKOS_LAMBDA(int i) {
            int p0 = i + 0 * xsz;
            double h0 = H(p0);
            double z0 = DEM(p0);
            if(z0 == nodata || h0 < 0.001) {
                Qy(i) = 0.0;
            } else {
                if(h0 > 0.001) {
                    double Q = -h0 * sqrt(g * h0) * dx * 0.5;
                    Qy(i) = Q;
                } else {
                    Qy(i) = 0.0;
                }
            }
        });
    } else {
        Kokkos::parallel_for("BCs_North_CLOSED", xsz + 1, KOKKOS_LAMBDA(int i) {
            Qy(i) = 0.0;
        });
    }

    // 南边界 (j=ysz)
    if(south_free) {
        Kokkos::parallel_for("BCs_South_FREE", xsz, KOKKOS_LAMBDA(int i) {
            int p0 = i + (ysz-1) * xsz;
            double h0 = H(p0);
            double z0 = DEM(p0);
            if(z0 == nodata || h0 < 0.001) {
                Qy(i + ysz * (xsz+1)) = 0.0;
            } else {
                if(h0 > 0.001) {
                    double Q = h0 * sqrt(g * h0) * dx * 0.5;
                    Qy(i + ysz * (xsz+1)) = Q;
                } else {
                    Qy(i + ysz * (xsz+1)) = 0.0;
                }
            }
        });
    } else {
        Kokkos::parallel_for("BCs_South_CLOSED", xsz + 1, KOKKOS_LAMBDA(int i) {
            Qy(i + ysz * (xsz+1)) = 0.0;
        });
    }

    Kokkos::fence();
}

// GPU点边界条件应用
void ApplyPointBCs_GPU(RealView& H, IntView& bc_idx, IntView& bc_type, RealView& bc_value,
                       int num_bcs, int ncols, double dx, double dA, double Tstep) {
    Kokkos::parallel_for("ApplyPointBCs", num_bcs, KOKKOS_LAMBDA(int b) {
        int idx = bc_idx(b);
        int type = bc_type(b);
        double val = bc_value(b);

        if(type == 1) {
            // BC_FREE: 排水口
            H(idx) = 0.0;
        }
        else if(type == 2 || type == 3) {
            // BC_HFIX / BC_HVAR: 固定/时变水位
            H(idx) = val;
        }
    });

    Kokkos::fence();
}

// GPU水深更新
void UpdateH_GPU(RealView& H, RealView& Qx, RealView& Qy, int xsz, int ysz, double dA, double Tstep) {
    Kokkos::parallel_for("UpdateH", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{xsz,ysz}),
        KOKKOS_LAMBDA(int i, int j) {
            int idx = i + j * xsz;
            double Qx_in = Qx(i + j*(xsz+1)), Qx_out = Qx((i+1) + j*(xsz+1));
            double Qy_in = Qy(i + j*(xsz+1)), Qy_out = Qy(i + (j+1)*(xsz+1));
            H(idx) += Tstep * (Qx_in - Qx_out + Qy_in - Qy_out) / dA;
            if(H(idx) < 0.0) H(idx) = 0.0;
        });
    Kokkos::fence();
}

// GPU流速计算
void ComputeVelocity_GPU(RealView& Vx_out, RealView& Vy_out,
                         RealView& H, RealView& DEM, RealView& Qx, RealView& Qy,
                         int xsz, int ysz, double dx, double nodata) {
    Kokkos::parallel_for("ComputeVx", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{xsz,ysz}),
        KOKKOS_LAMBDA(int i, int j) {
            int idx = i + j * xsz;
            double z = DEM(idx);
            double h = H(idx);

            if(z == nodata || h < 0.001) {
                Vx_out(idx) = 0.0;
                return;
            }

            double Qx_left = Qx(i + j*(xsz+1));
            double Qx_right = Qx((i+1) + j*(xsz+1));

            double hflow = h;
            if(hflow < 0.001) hflow = 0.001;

            double Vx_left = (fabs(Qx_left) > 1e-10) ? Qx_left / dx / hflow : 0.0;
            double Vx_right = (fabs(Qx_right) > 1e-10) ? Qx_right / dx / hflow : 0.0;

            if(fabs(Vx_left) > fabs(Vx_right)) {
                Vx_out(idx) = Vx_left;
            } else {
                Vx_out(idx) = Vx_right;
            }
        });

    Kokkos::parallel_for("ComputeVy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{xsz,ysz}),
        KOKKOS_LAMBDA(int i, int j) {
            int idx = i + j * xsz;
            double z = DEM(idx);
            double h = H(idx);

            if(z == nodata || h < 0.001) {
                Vy_out(idx) = 0.0;
                return;
            }

            double Qy_top = Qy(i + j*(xsz+1));
            double Qy_bottom = Qy(i + (j+1)*(xsz+1));

            double hflow = h;
            if(hflow < 0.001) hflow = 0.001;

            double Vy_top = (fabs(Qy_top) > 1e-10) ? Qy_top / dx / hflow : 0.0;
            double Vy_bottom = (fabs(Qy_bottom) > 1e-10) ? Qy_bottom / dx / hflow : 0.0;

            if(fabs(Vy_top) > fabs(Vy_bottom)) {
                Vy_out(idx) = Vy_top;
            } else {
                Vy_out(idx) = Vy_bottom;
            }
        });

    Kokkos::fence();
}

// GPU降雨计算
void Rainfall_GPU(RealView& H, RealView& DEM, int xsz, int ysz,
                  double rain_rate, double Tstep, double nodata) {
    if(rain_rate <= 0.0) return;

    double rain_depth = rain_rate * Tstep;

    Kokkos::parallel_for("Rainfall", xsz * ysz, KOKKOS_LAMBDA(int idx) {
        if(DEM(idx) != nodata) {
            H(idx) += rain_depth;
            if(H(idx) < 0.0) H(idx) = 0.0;
        }
    });
    Kokkos::fence();
}
