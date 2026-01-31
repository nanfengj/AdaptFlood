// io.cpp - 文件读写函数实现
#include "io.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

// 读取.par文件
bool read_par(const char* filename, Parameters& p) {
    std::ifstream file(filename);
    if(!file.is_open()) {
        printf("Error: Cannot open parameter file %s\n", filename);
        return false;
    }

    std::string line;
    while(std::getline(file, line)) {
        // 跳过注释和空行
        if(line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string key, value;
        if(iss >> key) {
            // 尝试读取value（可能没有）
            iss >> value;

            if(key == "DEMfile") p.dem_file = value;
            else if(key == "bcifile") p.bci_file = value;
            else if(key == "bdyfile") p.bdy_file = value;
            else if(key == "dirroot") p.output_dir = value;
            else if(key == "sim_time") p.sim_time = std::stod(value);
            else if(key == "saveint") p.save_int = std::stod(value);
            else if(key == "massint") p.mass_int = std::stod(value);
            else if(key == "initial_tstep") p.init_tstep = std::stod(value);
            else if(key == "cfl") p.cfl = std::stod(value);
            else if(key == "fpfric") p.fpn = std::stod(value);
            else if(key == "depth_thresh") p.depth_thresh = std::stod(value);
            else if(key == "rainfile") p.rain_file = value;
            else if(key == "stagefile") p.stage_file = value;
            else if(key == "stageint") p.stage_int = std::stod(value);
            else if(key == "output_format") p.output_format = std::stoi(value);
            else if(key == "voutput") p.voutput = true;
            else if(key == "async_output") p.async_output = (std::stoi(value) != 0);
        }
    }
    return true;
}

// 读取Stage监测点文件
bool read_stage_file(const char* filename, double xll, double yll, int nrows, double cellsize,
                     std::vector<StagePoint>& stages) {
    std::ifstream file(filename);
    if(!file.is_open()) {
        printf("Warning: Cannot open stage file %s, stage output disabled\n", filename);
        return false;
    }

    printf("Loading stage points: %s\n", filename);
    double tly = yll + nrows * cellsize;

    std::string line;
    int count = 0;
    while(std::getline(file, line)) {
        if(line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        StagePoint sp;
        if(iss >> sp.x >> sp.y) {
            iss >> sp.name;
            if(sp.name.empty()) {
                sp.name = "Stage" + std::to_string(count + 1);
            }

            sp.grid_x = (int)((sp.x - xll) / cellsize);
            sp.grid_y = (int)((tly - sp.y) / cellsize);

            stages.push_back(sp);
            printf("  %s at (%.1f, %.1f) -> grid (%d, %d)\n",
                   sp.name.c_str(), sp.x, sp.y, sp.grid_x, sp.grid_y);
            count++;
        }
    }
    printf("Total %d stage points loaded\n", count);
    return count > 0;
}

// 降雨率插值
double interpolate_rain(const RainfallData& rain, double t) {
    if(!rain.enabled || rain.times.empty()) return 0.0;
    if(t <= rain.times[0]) return rain.rates[0];
    if(t >= rain.times.back()) return rain.rates.back();

    for(size_t i = 0; i < rain.times.size() - 1; i++) {
        if(t >= rain.times[i] && t < rain.times[i+1]) {
            double ratio = (t - rain.times[i]) / (rain.times[i+1] - rain.times[i]);
            return rain.rates[i] + ratio * (rain.rates[i+1] - rain.rates[i]);
        }
    }
    return rain.rates.back();
}

// 读取降雨文件
bool read_rain(const char* filename, RainfallData& rain) {
    std::ifstream file(filename);
    if(!file.is_open()) {
        printf("Warning: Cannot open rain file %s, rainfall disabled\n", filename);
        rain.enabled = false;
        return false;
    }

    std::string line;
    std::getline(file, line); // 跳过标题行

    while(std::getline(file, line)) {
        if(line.empty() || line[0] == '#') continue;
        double rate, time;
        if(sscanf(line.c_str(), "%lf %lf", &rate, &time) == 2) {
            rain.times.push_back(time);
            rain.rates.push_back(rate);
        }
    }

    if(!rain.times.empty()) {
        rain.enabled = true;
        printf("Rainfall: %zu data points, rate=%.2e-%.2e m/s\n",
               rain.times.size(), rain.rates[0], rain.rates.back());
    }
    return rain.enabled;
}

// 读取DEM文件
bool read_dem(const char* filename, int& ncols, int& nrows, double& xll, double& yll,
              double& cellsize, double& nodata, std::vector<double>& data) {
    FILE* fp = fopen(filename, "r");
    if(!fp) { printf("Error: Cannot open DEM file %s\n", filename); return false; }

    char key[64];
    fscanf(fp, "%s %d", key, &ncols);
    fscanf(fp, "%s %d", key, &nrows);
    fscanf(fp, "%s %lf", key, &xll);
    fscanf(fp, "%s %lf", key, &yll);
    fscanf(fp, "%s %lf", key, &cellsize);
    fscanf(fp, "%s %lf", key, &nodata);

    printf("DEM: %d x %d, cellsize=%.1f\n", ncols, nrows, cellsize);
    fflush(stdout);

    printf("Reading DEM data (%d cells)...\n", ncols * nrows);
    fflush(stdout);

    data.resize(ncols * nrows);
    for(int j = 0; j < nrows; j++) {
        for(int i = 0; i < ncols; i++) {
            fscanf(fp, "%lf", &data[i + j * ncols]);
        }
    }
    fclose(fp);
    printf("DEM loaded.\n");
    fflush(stdout);
    return true;
}

// 读取bci文件（边界条件定义）
bool read_bci(const char* filename, double xll, double yll, int ncols, int nrows,
              double cellsize, std::vector<PointBoundary>& point_bcs,
              std::vector<LineBoundary>& line_bcs) {
    FILE* fp = fopen(filename, "r");
    if(!fp) { printf("Error: Cannot open BCI file %s\n", filename); return false; }

    double tly = yll + nrows * cellsize;

    char line[256];
    while(fgets(line, sizeof(line), fp)) {
        if(line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

        char type = line[0];

        if(type == 'P') {
            double x, y;
            char bctype[16], extra[64];
            int n = sscanf(line, "%c %lf %lf %s %s", &type, &x, &y, bctype, extra);

            if(n >= 4) {
                PointBoundary pb;
                pb.x = (int)((x - xll) / cellsize);
                pb.y = (int)((tly - y) / cellsize);
                pb.idx = pb.x + pb.y * ncols;
                pb.value = 0.0;

                if(strcmp(bctype, "QVAR") == 0 && n >= 5) {
                    pb.type = BC_QVAR;
                    pb.name = extra;
                    printf("Point BC: QVAR '%s' at grid (%d, %d)\n", pb.name.c_str(), pb.x, pb.y);
                }
                else if(strcmp(bctype, "QFIX") == 0 && n >= 5) {
                    pb.type = BC_QFIX;
                    pb.value = atof(extra);
                    printf("Point BC: QFIX %.4f at grid (%d, %d)\n", pb.value, pb.x, pb.y);
                }
                else if(strcmp(bctype, "FREE") == 0) {
                    pb.type = BC_FREE;
                    printf("Point BC: FREE (drainage) at grid (%d, %d)\n", pb.x, pb.y);
                }
                else if(strcmp(bctype, "HFIX") == 0 && n >= 5) {
                    pb.type = BC_HFIX;
                    pb.value = atof(extra);
                    printf("Point BC: HFIX %.4f at grid (%d, %d)\n", pb.value, pb.x, pb.y);
                }
                else if(strcmp(bctype, "HVAR") == 0 && n >= 5) {
                    pb.type = BC_HVAR;
                    pb.name = extra;
                    printf("Point BC: HVAR '%s' at grid (%d, %d)\n", pb.name.c_str(), pb.x, pb.y);
                }
                else {
                    printf("Warning: Unknown point BC type '%s', skipped\n", bctype);
                    continue;
                }
                point_bcs.push_back(pb);
            }
        }
        else if(type == 'N' || type == 'S' || type == 'E' || type == 'W') {
            int start_idx, end_idx;
            char bctype[16];
            int n = sscanf(line, "%c %d %d %s", &type, &start_idx, &end_idx, bctype);

            LineBoundary lb;
            lb.side = type;
            lb.type = BC_CLOSED;

            if(n >= 4 && strcmp(bctype, "FREE") == 0) {
                lb.type = BC_FREE;
                lb.start_idx = start_idx;
                lb.end_idx = end_idx;
                printf("Line BC: %c FREE from %d to %d\n", type, start_idx, end_idx);
            }
            else if(n >= 2) {
                char bctype2[16];
                if(sscanf(line, "%c %s", &type, bctype2) == 2 && strcmp(bctype2, "FREE") == 0) {
                    lb.type = BC_FREE;
                    lb.start_idx = 0;
                    lb.end_idx = (type == 'N' || type == 'S') ? ncols - 1 : nrows - 1;
                    printf("Line BC: %c FREE (entire edge)\n", type);
                }
            }

            if(lb.type == BC_FREE) {
                line_bcs.push_back(lb);
            }
        }
    }
    fclose(fp);
    printf("Total %zu point boundaries, %zu line boundaries\n", point_bcs.size(), line_bcs.size());
    return true;
}

// 读取bdy文件（时变边界数据）
bool read_bdy(const char* filename, std::vector<PointBoundary>& point_bcs) {
    std::ifstream file(filename);
    if(!file.is_open()) { printf("Error: Cannot open BDY file %s\n", filename); return false; }

    std::string line;
    std::getline(file, line);

    while(std::getline(file, line)) {
        if(line.empty() || line[0] == '#') continue;

        for(auto& pb : point_bcs) {
            if(pb.type != BC_QVAR && pb.type != BC_HVAR) continue;

            if(line.find(pb.name) == 0 &&
               (line.length() == pb.name.length() ||
                line[pb.name.length()] == ' ' ||
                line[pb.name.length()] == '\n' ||
                line[pb.name.length()] == '\r')) {
                std::getline(file, line);
                int n;
                sscanf(line.c_str(), "%d", &n);

                for(int i = 0; i < n; i++) {
                    std::getline(file, line);
                    double val, time;
                    sscanf(line.c_str(), "%lf %lf", &val, &time);
                    pb.times.push_back(time);
                    pb.values.push_back(val);
                }

                const char* type_str = (pb.type == BC_QVAR) ? "QVAR" : "HVAR";
                printf("  %s (%s): %zu data points, val=%.4f-%.4f\n",
                       pb.name.c_str(), type_str, pb.times.size(),
                       pb.values[0], pb.values.back());
                break;
            }
        }
    }
    return true;
}

// 线性插值获取当前时刻值
double interpolate_bc_value(const PointBoundary& pb, double t) {
    if(pb.times.empty()) return pb.value;
    if(t <= pb.times[0]) return pb.values[0];
    if(t >= pb.times.back()) return pb.values.back();

    for(size_t i = 0; i < pb.times.size() - 1; i++) {
        if(t >= pb.times[i] && t < pb.times[i+1]) {
            double ratio = (t - pb.times[i]) / (pb.times[i+1] - pb.times[i]);
            return pb.values[i] + ratio * (pb.values[i+1] - pb.values[i]);
        }
    }
    return pb.values.back();
}

// 写ASC输出文件
void write_asc(const char* filename, RealView& H, int ncols, int nrows,
               double xll, double yll, double cellsize) {
    auto H_host = Kokkos::create_mirror_view(H);
    Kokkos::deep_copy(H_host, H);

    FILE* fp = fopen(filename, "w");
    fprintf(fp, "ncols         %d\n", ncols);
    fprintf(fp, "nrows         %d\n", nrows);
    fprintf(fp, "xllcorner     %.1f\n", xll);
    fprintf(fp, "yllcorner     %.1f\n", yll);
    fprintf(fp, "cellsize      %.1f\n", cellsize);
    fprintf(fp, "NODATA_value  -9999\n");

    for(int j = 0; j < nrows; j++) {
        for(int i = 0; i < ncols; i++) {
            fprintf(fp, "%.4f ", H_host(i + j * ncols));
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}
