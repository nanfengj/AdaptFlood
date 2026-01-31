// main.cpp - LISFLOOD-GPU 主程序
// GPU版本LISFLOOD洪水模拟
// 支持读取 .par 参数文件
// 支持NetCDF输出和异步I/O
// 用法: ./lisflood_gpu <parfile>

#include "types.hpp"
#include "io.hpp"
#include "kernels.hpp"
#include "async_output.hpp"

#include <cstdio>
#include <chrono>
#include <sys/stat.h>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        printf("=== LISFLOOD-GPU Flood Simulation ===\n");
        printf("Execution space: %s\n\n", typeid(Kokkos::DefaultExecutionSpace).name());
        fflush(stdout);

        // 读取参数文件
        Parameters par;
        if(argc > 1) {
            printf("Reading parameter file: %s\n", argv[1]);
            if(!read_par(argv[1], par)) return 1;
        } else {
            printf("Usage: %s <parfile>\n", argv[0]);
            printf("Using default parameters...\n\n");
        }

        printf("Parameters:\n");
        printf("  DEM file:    %s\n", par.dem_file.c_str());
        printf("  BCI file:    %s\n", par.bci_file.c_str());
        printf("  BDY file:    %s\n", par.bdy_file.c_str());
        printf("  Output dir:  %s\n", par.output_dir.c_str());
        printf("  Sim time:    %.0f s\n", par.sim_time);
        printf("  Save int:    %.0f s\n", par.save_int);
        printf("  CFL:         %.2f (adaptive timestep)\n", par.cfl);
        printf("  Manning n:   %.4f\n", par.fpn);
        const char* fmt_str[] = {"ASC", "NetCDF", "Both"};
        printf("  Output fmt:  %s\n", fmt_str[par.output_format]);
        printf("  Velocity:    %s\n", par.voutput ? "enabled" : "disabled");
        printf("  Async I/O:   %s\n", par.async_output ? "enabled" : "disabled");
        printf("\n");
        fflush(stdout);

        // 读取DEM
        int ncols, nrows;
        double xll, yll, cellsize, nodata;
        std::vector<double> dem_data;
        if(!read_dem(par.dem_file.c_str(), ncols, nrows, xll, yll, cellsize, nodata, dem_data)) return 1;

        // 读取边界条件
        std::vector<PointBoundary> point_bcs;
        std::vector<LineBoundary> line_bcs;
        if(!read_bci(par.bci_file.c_str(), xll, yll, ncols, nrows, cellsize, point_bcs, line_bcs)) return 1;
        if(!read_bdy(par.bdy_file.c_str(), point_bcs)) return 1;

        // 解析线边界标志
        bool west_free = false, east_free = false, north_free = false, south_free = false;
        for(const auto& lb : line_bcs) {
            if(lb.type == BC_FREE) {
                if(lb.side == 'W') west_free = true;
                else if(lb.side == 'E') east_free = true;
                else if(lb.side == 'N') north_free = true;
                else if(lb.side == 'S') south_free = true;
            }
        }
        if(west_free || east_free || north_free || south_free) {
            printf("Line FREE boundaries: W=%d E=%d N=%d S=%d\n",
                   west_free, east_free, north_free, south_free);
        }

        // 分离流量边界和其他点边界
        std::vector<PointBoundary> flow_bcs;
        std::vector<PointBoundary> other_bcs;
        for(const auto& pb : point_bcs) {
            if(pb.type == BC_QVAR || pb.type == BC_QFIX) {
                flow_bcs.push_back(pb);
            } else {
                other_bcs.push_back(pb);
            }
        }
        printf("Flow boundaries: %zu, Other boundaries: %zu\n", flow_bcs.size(), other_bcs.size());

        // 读取降雨（可选）
        RainfallData rainfall;
        if(!par.rain_file.empty()) {
            read_rain(par.rain_file.c_str(), rainfall);
        }

        // 读取Stage监测点（可选）
        std::vector<StagePoint> stages;
        FILE* stage_fp = nullptr;
        if(!par.stage_file.empty()) {
            read_stage_file(par.stage_file.c_str(), xll, yll, nrows, cellsize, stages);
            if(!stages.empty()) {
                char stage_filename[256];
                sprintf(stage_filename, "%s/stage.csv", par.output_dir.c_str());
                stage_fp = fopen(stage_filename, "w");
                if(stage_fp) {
                    fprintf(stage_fp, "Time(s)");
                    for(const auto& sp : stages) {
                        fprintf(stage_fp, ",%s", sp.name.c_str());
                    }
                    fprintf(stage_fp, "\n");
                    printf("Stage output file: %s\n", stage_filename);
                }
            }
        }

        // 参数设置
        double dx = cellsize, dy = cellsize, dA = dx * dy;
        double FPn = par.fpn;
        double DepthThresh = par.depth_thresh;
        double MaxHflow = 10.0;
        double dhlin = 0.01;
        double Qlimfact = 1.0;
        double InitTstep = par.init_tstep;
        double cfl = par.cfl;
        double g = 9.81;
        double theta = 1.0;
        double sim_time = par.sim_time;
        double save_int = par.save_int;
        double mass_int = par.mass_int;

        // 创建输出目录
        mkdir(par.output_dir.c_str(), 0755);

        // 分配GPU数组
        RealView H("H", ncols * nrows);
        RealView DEM("DEM", ncols * nrows);
        RealView Qx("Qx", (ncols+1) * (nrows+1));
        RealView Qy("Qy", (ncols+1) * (nrows+1));
        RealView Qxold("Qxold", (ncols+1) * (nrows+1));
        RealView Qyold("Qyold", (ncols+1) * (nrows+1));

        RealView Vx("Vx", par.voutput ? ncols * nrows : 1);
        RealView Vy("Vy", par.voutput ? ncols * nrows : 1);

        // 初始化异步输出管理器
        AsyncOutputManager async_output;
        if(par.async_output) {
            async_output.init(ncols, nrows, xll, yll, cellsize, nodata,
                             par.output_dir, par.output_format, par.voutput);
        }

        // 复制DEM到GPU
        auto DEM_host = Kokkos::create_mirror_view(DEM);
        for(int i = 0; i < ncols * nrows; i++) DEM_host(i) = dem_data[i];
        Kokkos::deep_copy(DEM, DEM_host);

        // 初始化数组
        Kokkos::parallel_for("InitH", ncols*nrows, KOKKOS_LAMBDA(int i) { H(i) = 0.0; });
        Kokkos::parallel_for("InitQx", (ncols+1)*(nrows+1), KOKKOS_LAMBDA(int i) { Qx(i) = 0.0; });
        Kokkos::parallel_for("InitQy", (ncols+1)*(nrows+1), KOKKOS_LAMBDA(int i) { Qy(i) = 0.0; });
        Kokkos::parallel_for("InitQxold", (ncols+1)*(nrows+1), KOKKOS_LAMBDA(int i) { Qxold(i) = 0.0; });
        Kokkos::parallel_for("InitQyold", (ncols+1)*(nrows+1), KOKKOS_LAMBDA(int i) { Qyold(i) = 0.0; });
        Kokkos::fence();

        printf("\nStarting simulation: %.0f seconds, output every %.0f seconds\n\n", sim_time, save_int);
        fflush(stdout);

        // 预分配流量边界数组
        int num_flow_bcs = flow_bcs.size();
        IntView ps_idx("ps_idx", num_flow_bcs > 0 ? num_flow_bcs : 1);
        RealView ps_flow("ps_flow", num_flow_bcs > 0 ? num_flow_bcs : 1);
        auto ps_idx_host = Kokkos::create_mirror_view(ps_idx);
        auto ps_flow_host = Kokkos::create_mirror_view(ps_flow);
        for(int s = 0; s < num_flow_bcs; s++) {
            ps_idx_host(s) = flow_bcs[s].idx;
        }
        Kokkos::deep_copy(ps_idx, ps_idx_host);

        // 预分配其他点边界数组
        int num_other_bcs = other_bcs.size();
        IntView bc_idx("bc_idx", num_other_bcs > 0 ? num_other_bcs : 1);
        IntView bc_type("bc_type", num_other_bcs > 0 ? num_other_bcs : 1);
        RealView bc_value("bc_value", num_other_bcs > 0 ? num_other_bcs : 1);
        auto bc_idx_host = Kokkos::create_mirror_view(bc_idx);
        auto bc_type_host = Kokkos::create_mirror_view(bc_type);
        auto bc_value_host = Kokkos::create_mirror_view(bc_value);
        for(int b = 0; b < num_other_bcs; b++) {
            bc_idx_host(b) = other_bcs[b].idx;
            if(other_bcs[b].type == BC_FREE) bc_type_host(b) = 1;
            else if(other_bcs[b].type == BC_HFIX) bc_type_host(b) = 2;
            else if(other_bcs[b].type == BC_HVAR) bc_type_host(b) = 3;
            bc_value_host(b) = other_bcs[b].value;
        }
        Kokkos::deep_copy(bc_idx, bc_idx_host);
        Kokkos::deep_copy(bc_type, bc_type_host);
        Kokkos::deep_copy(bc_value, bc_value_host);

        // 时间循环
        double t = 0.0;
        int step = 0;
        double next_save = save_int;
        double next_mass = mass_int;
        double next_stage = par.stage_int;
        int save_num = 0;

        auto wall_start = std::chrono::high_resolution_clock::now();

        // 保存初始状态
        if(par.async_output && async_output.is_running()) {
            OutputData out_data;
            out_data.time = 0.0;
            out_data.save_num = save_num;
            out_data.H.resize(ncols * nrows, 0.0);
            if(par.voutput) {
                out_data.Vx.resize(ncols * nrows, 0.0);
                out_data.Vy.resize(ncols * nrows, 0.0);
            }
            async_output.enqueue(out_data);
            printf("Queued: t=0s (initial state)\n");
        } else {
            char filename[256];
            sprintf(filename, "%s/H_%04d.asc", par.output_dir.c_str(), save_num);
            write_asc(filename, H, ncols, nrows, xll, yll, cellsize);
            printf("Saved: %s (initial state)\n", filename);
        }
        fflush(stdout);
        save_num++;

        while(t < sim_time) {
            double Tstep = InitTstep;

            // 1. 计算流量
            FloodplainQ_GPU(H, DEM, Qx, Qy, Qxold, Qyold, ncols, nrows, dx, dy, FPn,
                           DepthThresh, MaxHflow, dhlin, Tstep, Qlimfact, dA,
                           cfl, g, InitTstep, theta, nodata);

            // 2. 边界条件处理
            BCs_GPU(Qx, Qy, H, DEM, ncols, nrows, dx, g, Tstep, nodata,
                    west_free, east_free, north_free, south_free);

            if(step < 10 || step % 100 == 0) {
                printf("Step %d, t=%.2fs, dt=%.4fs\n", step, t, Tstep);
                fflush(stdout);
            }

            // 3. 添加流量边界
            if(num_flow_bcs > 0) {
                for(int s = 0; s < num_flow_bcs; s++) {
                    double flow_val;
                    if(flow_bcs[s].type == BC_QFIX) {
                        flow_val = flow_bcs[s].value;
                    } else {
                        flow_val = interpolate_bc_value(flow_bcs[s], t);
                    }
                    ps_flow_host(s) = flow_val * dx * Tstep / dA;
                }
                Kokkos::deep_copy(ps_flow, ps_flow_host);
                Kokkos::parallel_for("AddFlowBCs", num_flow_bcs, KOKKOS_LAMBDA(int s) {
                    int idx = ps_idx(s);
                    H(idx) += ps_flow(s);
                    if(H(idx) < 0.0) H(idx) = 0.0;
                });
                Kokkos::fence();
            }

            // 4. 添加降雨
            if(rainfall.enabled) {
                double rain_rate = interpolate_rain(rainfall, t);
                Rainfall_GPU(H, DEM, ncols, nrows, rain_rate, Tstep, nodata);
            }

            // 5. 更新水深
            UpdateH_GPU(H, Qx, Qy, ncols, nrows, dA, Tstep);

            // 6. 应用点边界条件
            if(num_other_bcs > 0) {
                for(int b = 0; b < num_other_bcs; b++) {
                    if(other_bcs[b].type == BC_HVAR) {
                        bc_value_host(b) = interpolate_bc_value(other_bcs[b], t);
                    }
                }
                Kokkos::deep_copy(bc_value, bc_value_host);
                ApplyPointBCs_GPU(H, bc_idx, bc_type, bc_value, num_other_bcs, ncols, dx, dA, Tstep);
            }

            // 7. 更新Qold数组
            UpdateQs_GPU(Qx, Qy, Qxold, Qyold, ncols, nrows, dx);

            t += Tstep;
            step++;

            // 质量报告
            if(t >= next_mass) {
                double total_vol = 0.0;
                Kokkos::parallel_reduce("Vol", ncols*nrows,
                    KOKKOS_LAMBDA(int i, double& v) { v += H(i) * dA; }, total_vol);

                double flood_area = 0.0;
                Kokkos::parallel_reduce("Area", ncols*nrows,
                    KOKKOS_LAMBDA(int i, double& a) { if(H(i)>0.01) a += dA; }, flood_area);

                printf("t=%.0fs: Tstep=%.2f, Vol=%.0fm3, Area=%.0fm2\n",
                       t, Tstep, total_vol, flood_area);
                fflush(stdout);
                next_mass += mass_int;
            }

            // Stage监测点输出
            if(stage_fp && t >= next_stage) {
                auto H_host = Kokkos::create_mirror_view(H);
                Kokkos::deep_copy(H_host, H);
                fprintf(stage_fp, "%.1f", t);
                for(const auto& sp : stages) {
                    int idx = sp.grid_x + sp.grid_y * ncols;
                    if(idx >= 0 && idx < ncols * nrows) {
                        fprintf(stage_fp, ",%.4f", H_host(idx));
                    } else {
                        fprintf(stage_fp, ",NaN");
                    }
                }
                fprintf(stage_fp, "\n");
                fflush(stage_fp);
                next_stage += par.stage_int;
            }

            // 保存输出
            if(t >= next_save) {
                if(par.async_output && async_output.is_running()) {
                    OutputData out_data;
                    out_data.time = t;
                    out_data.save_num = save_num;

                    auto H_host = Kokkos::create_mirror_view(H);
                    Kokkos::deep_copy(H_host, H);
                    out_data.H.resize(ncols * nrows);
                    for(int i = 0; i < ncols * nrows; i++) {
                        out_data.H[i] = H_host(i);
                    }

                    if(par.voutput) {
                        ComputeVelocity_GPU(Vx, Vy, H, DEM, Qx, Qy, ncols, nrows, dx, nodata);
                        auto Vx_host = Kokkos::create_mirror_view(Vx);
                        auto Vy_host = Kokkos::create_mirror_view(Vy);
                        Kokkos::deep_copy(Vx_host, Vx);
                        Kokkos::deep_copy(Vy_host, Vy);
                        out_data.Vx.resize(ncols * nrows);
                        out_data.Vy.resize(ncols * nrows);
                        for(int i = 0; i < ncols * nrows; i++) {
                            out_data.Vx[i] = Vx_host(i);
                            out_data.Vy[i] = Vy_host(i);
                        }
                    }

                    async_output.enqueue(out_data);
                    printf("Queued: t=%.0fs (save_num=%d)\n", t, save_num);
                } else {
                    char filename[256];
                    sprintf(filename, "%s/H_%04d.asc", par.output_dir.c_str(), save_num);
                    write_asc(filename, H, ncols, nrows, xll, yll, cellsize);
                    printf("Saved: %s\n", filename);
                }
                fflush(stdout);
                save_num++;
                next_save += save_int;
            }
        }

        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_seconds = std::chrono::duration<double>(wall_end - wall_start).count();

        if(par.async_output && async_output.is_running()) {
            printf("\nWaiting for async output to complete...\n");
            async_output.finish();
            printf("Async output finished.\n");
        }

        // 保存最终状态
        char final_file[256];
        sprintf(final_file, "%s/H_final.asc", par.output_dir.c_str());
        write_asc(final_file, H, ncols, nrows, xll, yll, cellsize);
        printf("Saved: %s\n", final_file);

        if(par.voutput) {
            ComputeVelocity_GPU(Vx, Vy, H, DEM, Qx, Qy, ncols, nrows, dx, nodata);
            sprintf(final_file, "%s/Vx_final.asc", par.output_dir.c_str());
            write_asc(final_file, Vx, ncols, nrows, xll, yll, cellsize);
            printf("Saved: %s\n", final_file);
            sprintf(final_file, "%s/Vy_final.asc", par.output_dir.c_str());
            write_asc(final_file, Vy, ncols, nrows, xll, yll, cellsize);
            printf("Saved: %s\n", final_file);
        }

        if(stage_fp) {
            fclose(stage_fp);
            printf("Stage output saved.\n");
        }

        double final_vol = 0.0;
        Kokkos::parallel_reduce("FinalVol", ncols*nrows,
            KOKKOS_LAMBDA(int i, double& v) { v += H(i) * dA; }, final_vol);

        printf("\n=== Simulation Complete ===\n");
        printf("Simulation time: %.0f s, Steps: %d\n", t, step);
        printf("Wall clock time: %.2f s (%.2f min)\n", wall_seconds, wall_seconds/60.0);
        printf("Final volume: %.0f m3\n", final_vol);
        printf("Output saved to %s/\n", par.output_dir.c_str());
        if(par.output_format >= 1) {
            printf("NetCDF output: %s/output.nc\n", par.output_dir.c_str());
        }
    }
    Kokkos::finalize();
    return 0;
}
