// lisflood_gpu.cpp - GPU版本LISFLOOD点源扩散
// 支持读取 .par 参数文件
// 支持NetCDF输出和异步I/O
// 用法: ./lisflood_gpu <parfile>

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <sys/stat.h>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

#ifdef USE_NETCDF
#include <netcdf.h>
#define NC_CHECK(e) { int _nc_err = (e); if(_nc_err != NC_NOERR) { \
    printf("NetCDF error: %s\n", nc_strerror(_nc_err)); } }
#endif

using RealView = Kokkos::View<double*, Kokkos::CudaUVMSpace>;
using IntView = Kokkos::View<int*, Kokkos::CudaUVMSpace>;

// 辅助函数
KOKKOS_INLINE_FUNCTION double getmax(double a, double b) { return (a > b) ? a : b; }
KOKKOS_INLINE_FUNCTION double getmin(double a, double b) { return (a < b) ? a : b; }

// 参数结构
struct Parameters {
    std::string dem_file = "dem.ascii";
    std::string bci_file = "bci.bci";
    std::string bdy_file = "bdy.bdy";
    std::string rain_file = "";         // 降雨文件（可选）
    std::string stage_file = "";        // Stage监测点文件（可选）
    std::string output_dir = "results";
    double sim_time = 3600.0;      // 1小时
    double save_int = 600.0;       // 10分钟
    double mass_int = 60.0;        // 1分钟
    double stage_int = 60.0;       // Stage输出间隔（秒）
    double init_tstep = 10.0;
    double cfl = 0.7;              // CFL系数（自适应时间步）
    double fpn = 0.035;            // Manning系数
    double depth_thresh = 0.001;
    double nodata = -9999.0;       // NoData值
    // 输出格式和流速选项
    int output_format = 0;         // 0=ASC, 1=NetCDF, 2=Both
    bool voutput = false;          // 是否输出流速
    bool async_output = true;      // 是否异步输出
};

// 边界条件类型
enum BoundaryType {
    BC_CLOSED = 0,  // 闭合边界（默认）
    BC_FREE = 1,    // 自由出流边界
    BC_HFIX = 2,    // 固定水位边界
    BC_HVAR = 3,    // 时变水位边界
    BC_QFIX = 4,    // 固定流量边界
    BC_QVAR = 5     // 时变流量边界（点源）
};

// 边界条件结构
struct BoundaryCondition {
    char side;          // 'N','S','E','W' 或 'P'(点)
    int start_idx;      // 起始索引
    int end_idx;        // 结束索引
    BoundaryType type;  // 边界类型
    double value;       // 固定值（HFIX/QFIX）
    std::string name;   // bdy文件中的名称（HVAR/QVAR）
};

// Stage监测点结构
struct StagePoint {
    double x, y;        // 世界坐标
    int grid_x, grid_y; // 网格坐标
    std::string name;   // 名称（可选）
};

// 点边界结构（统一处理所有点边界类型）
struct PointBoundary {
    int x, y;                    // 网格坐标
    int idx;                     // 线性索引
    BoundaryType type;           // BC_QFIX, BC_QVAR, BC_FREE, BC_HFIX, BC_HVAR
    double value;                // 固定值（QFIX, HFIX用）
    std::string name;            // 名称（QVAR/HVAR用于匹配bdy文件）
    std::vector<double> times;   // 时间序列
    std::vector<double> values;  // 流量/水位值
};

// 线边界结构
struct LineBoundary {
    char side;                   // 'N', 'S', 'E', 'W'
    BoundaryType type;           // BC_CLOSED 或 BC_FREE
    int start_idx;               // 起始单元索引
    int end_idx;                 // 结束单元索引
};

// 异步输出数据结构
struct OutputData {
    double time;
    int save_num;
    std::vector<double> H;
    std::vector<double> Vx;
    std::vector<double> Vy;
};

// 异步输出管理器
class AsyncOutputManager {
private:
    std::queue<OutputData> output_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::thread writer_thread;
    std::atomic<bool> stop_flag{false};
    std::atomic<bool> running{false};

    // NetCDF文件信息
    int ncid = -1;
    int time_dimid, x_dimid, y_dimid;
    int time_varid, H_varid, Vx_varid, Vy_varid;
    size_t time_index = 0;

    // 网格信息
    int ncols, nrows;
    double xll, yll, cellsize, nodata_val;
    std::string output_dir;
    int output_format;
    bool voutput;

public:
    void init(int _ncols, int _nrows, double _xll, double _yll,
              double _cellsize, double _nodata, const std::string& _output_dir,
              int _output_format, bool _voutput) {
        ncols = _ncols;
        nrows = _nrows;
        xll = _xll;
        yll = _yll;
        cellsize = _cellsize;
        nodata_val = _nodata;
        output_dir = _output_dir;
        output_format = _output_format;
        voutput = _voutput;

#ifdef USE_NETCDF
        if(output_format >= 1) {
            // 创建NetCDF文件
            std::string nc_file = output_dir + "/output.nc";
            NC_CHECK(nc_create(nc_file.c_str(), NC_CLOBBER | NC_NETCDF4, &ncid));

            // 定义维度
            NC_CHECK(nc_def_dim(ncid, "time", NC_UNLIMITED, &time_dimid));
            NC_CHECK(nc_def_dim(ncid, "x", ncols, &x_dimid));
            NC_CHECK(nc_def_dim(ncid, "y", nrows, &y_dimid));

            // 定义时间变量
            NC_CHECK(nc_def_var(ncid, "time", NC_DOUBLE, 1, &time_dimid, &time_varid));
            nc_put_att_text(ncid, time_varid, "units", 7, "seconds");

            // 定义水深变量 H(time, y, x)
            int dims3d[3] = {time_dimid, y_dimid, x_dimid};
            NC_CHECK(nc_def_var(ncid, "H", NC_FLOAT, 3, dims3d, &H_varid));
            nc_put_att_text(ncid, H_varid, "units", 1, "m");
            nc_put_att_text(ncid, H_varid, "long_name", 11, "water_depth");
            float fill_val = (float)nodata_val;
            nc_put_att_float(ncid, H_varid, "_FillValue", NC_FLOAT, 1, &fill_val);

            // 定义流速变量（如果启用）
            if(voutput) {
                NC_CHECK(nc_def_var(ncid, "Vx", NC_FLOAT, 3, dims3d, &Vx_varid));
                nc_put_att_text(ncid, Vx_varid, "units", 3, "m/s");
                nc_put_att_text(ncid, Vx_varid, "long_name", 16, "x_direction_velocity");

                NC_CHECK(nc_def_var(ncid, "Vy", NC_FLOAT, 3, dims3d, &Vy_varid));
                nc_put_att_text(ncid, Vy_varid, "units", 3, "m/s");
                nc_put_att_text(ncid, Vy_varid, "long_name", 16, "y_direction_velocity");
            }

            // 全局属性
            nc_put_att_double(ncid, NC_GLOBAL, "xllcorner", NC_DOUBLE, 1, &xll);
            nc_put_att_double(ncid, NC_GLOBAL, "yllcorner", NC_DOUBLE, 1, &yll);
            nc_put_att_double(ncid, NC_GLOBAL, "cellsize", NC_DOUBLE, 1, &cellsize);

            NC_CHECK(nc_enddef(ncid));
            printf("NetCDF file created: %s\n", nc_file.c_str());
        }
#endif

        // 启动写入线程
        stop_flag = false;
        running = true;
        writer_thread = std::thread(&AsyncOutputManager::writer_loop, this);
    }

    void enqueue(const OutputData& data) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            output_queue.push(data);
        }
        cv.notify_one();
    }

    void finish() {
        stop_flag = true;
        cv.notify_one();
        if(writer_thread.joinable()) {
            writer_thread.join();
        }
#ifdef USE_NETCDF
        if(ncid >= 0) {
            nc_close(ncid);
            ncid = -1;
        }
#endif
        running = false;
    }

    bool is_running() const { return running; }

private:
    void writer_loop() {
        while(true) {
            OutputData data;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv.wait(lock, [this]{ return !output_queue.empty() || stop_flag; });

                if(output_queue.empty() && stop_flag) break;
                if(output_queue.empty()) continue;

                data = std::move(output_queue.front());
                output_queue.pop();
            }

            // 写入文件
            write_output(data);
        }

        // 处理剩余的输出
        while(!output_queue.empty()) {
            OutputData data = std::move(output_queue.front());
            output_queue.pop();
            write_output(data);
        }
    }

    void write_output(const OutputData& data) {
#ifdef USE_NETCDF
        if(output_format >= 1 && ncid >= 0) {
            printf("[AsyncIO] Writing t=%.0fs to NetCDF (time_index=%zu)...\n", data.time, time_index);
            fflush(stdout);

            // 写入时间
            size_t start1[1] = {time_index};
            size_t count1[1] = {1};
            NC_CHECK(nc_put_vara_double(ncid, time_varid, start1, count1, &data.time));

            // 写入水深（转换为float以节省空间）
            std::vector<float> H_float(data.H.size());
            for(size_t i = 0; i < data.H.size(); i++) {
                H_float[i] = (float)data.H[i];
            }
            size_t start3[3] = {time_index, 0, 0};
            size_t count3[3] = {1, (size_t)nrows, (size_t)ncols};
            NC_CHECK(nc_put_vara_float(ncid, H_varid, start3, count3, H_float.data()));

            // 写入流速（如果有）
            if(voutput && !data.Vx.empty()) {
                std::vector<float> Vx_float(data.Vx.size());
                std::vector<float> Vy_float(data.Vy.size());
                for(size_t i = 0; i < data.Vx.size(); i++) {
                    Vx_float[i] = (float)data.Vx[i];
                    Vy_float[i] = (float)data.Vy[i];
                }
                NC_CHECK(nc_put_vara_float(ncid, Vx_varid, start3, count3, Vx_float.data()));
                NC_CHECK(nc_put_vara_float(ncid, Vy_varid, start3, count3, Vy_float.data()));
            }

            nc_sync(ncid);  // 确保数据写入磁盘
            printf("[AsyncIO] Written t=%.0fs to NetCDF, synced.\n", data.time);
            fflush(stdout);
            time_index++;
        }
#endif

        // ASC输出（如果需要）
        if(output_format == 0 || output_format == 2) {
            char filename[256];
            sprintf(filename, "%s/H_%04d.asc", output_dir.c_str(), data.save_num);
            write_asc_file(filename, data.H);

            if(voutput && !data.Vx.empty()) {
                sprintf(filename, "%s/Vx_%04d.asc", output_dir.c_str(), data.save_num);
                write_asc_file(filename, data.Vx);
                sprintf(filename, "%s/Vy_%04d.asc", output_dir.c_str(), data.save_num);
                write_asc_file(filename, data.Vy);
            }
        }
    }

    void write_asc_file(const char* filename, const std::vector<double>& data) {
        FILE* fp = fopen(filename, "w");
        if(!fp) return;

        fprintf(fp, "ncols         %d\n", ncols);
        fprintf(fp, "nrows         %d\n", nrows);
        fprintf(fp, "xllcorner     %.6f\n", xll);
        fprintf(fp, "yllcorner     %.6f\n", yll);
        fprintf(fp, "cellsize      %.1f\n", cellsize);
        fprintf(fp, "NODATA_value  %.0f\n", nodata_val);

        for(int j = 0; j < nrows; j++) {
            for(int i = 0; i < ncols; i++) {
                fprintf(fp, "%.4f ", data[i + j * ncols]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
};

// 降雨时间序列结构
struct RainfallData {
    std::vector<double> times;     // 时间点
    std::vector<double> rates;     // 降雨率 (m/s)
    bool enabled = false;
};

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
            else if(key == "stagefile") p.stage_file = value;  // Stage监测点文件
            else if(key == "stageint") p.stage_int = std::stod(value);  // Stage输出间隔
            // 输出参数
            else if(key == "output_format") p.output_format = std::stoi(value);  // 0=ASC, 1=NetCDF, 2=Both
            else if(key == "voutput") p.voutput = true;  // 启用流速输出（无需值）
            else if(key == "async_output") p.async_output = (std::stoi(value) != 0);
        }
    }
    return true;
}

// 读取Stage监测点文件
// 格式：每行一个监测点，x y [name]
bool read_stage_file(const char* filename, double xll, double yll, int nrows, double cellsize,
                     std::vector<StagePoint>& stages) {
    std::ifstream file(filename);
    if(!file.is_open()) {
        printf("Warning: Cannot open stage file %s, stage output disabled\n", filename);
        return false;
    }

    printf("Loading stage points: %s\n", filename);
    double tly = yll + nrows * cellsize;  // 顶部y坐标

    std::string line;
    int count = 0;
    while(std::getline(file, line)) {
        if(line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        StagePoint sp;
        if(iss >> sp.x >> sp.y) {
            // 可选的名称
            iss >> sp.name;
            if(sp.name.empty()) {
                sp.name = "Stage" + std::to_string(count + 1);
            }

            // 转换为网格坐标（使用LISFLOOD坐标系统）
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

// 读取降雨文件（简化格式：时间 降雨率）
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
// 支持的格式：
//   P x y QVAR name     - 时变流量点（原有）
//   P x y QFIX value    - 固定流量点
//   P x y FREE          - 自由出流点（排水口）
//   P x y HFIX value    - 固定水位点
//   P x y HVAR name     - 时变水位点
//   N/S/E/W [start end] FREE  - 线边界自由出流
bool read_bci(const char* filename, double xll, double yll, int ncols, int nrows,
              double cellsize, std::vector<PointBoundary>& point_bcs,
              std::vector<LineBoundary>& line_bcs) {
    FILE* fp = fopen(filename, "r");
    if(!fp) { printf("Error: Cannot open BCI file %s\n", filename); return false; }

    double tly = yll + nrows * cellsize;  // 顶部y坐标（LISFLOOD原版）

    char line[256];
    while(fgets(line, sizeof(line), fp)) {
        // 跳过空行和注释
        if(line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

        char type = line[0];

        if(type == 'P') {
            // 点边界
            double x, y;
            char bctype[16], extra[64];
            int n = sscanf(line, "%c %lf %lf %s %s", &type, &x, &y, bctype, extra);

            if(n >= 4) {
                PointBoundary pb;
                // 坐标转换
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
            // 线边界
            int start_idx, end_idx;
            char bctype[16];
            int n = sscanf(line, "%c %d %d %s", &type, &start_idx, &end_idx, bctype);

            LineBoundary lb;
            lb.side = type;
            lb.type = BC_CLOSED;  // 默认闭合

            if(n >= 4 && strcmp(bctype, "FREE") == 0) {
                lb.type = BC_FREE;
                lb.start_idx = start_idx;
                lb.end_idx = end_idx;
                printf("Line BC: %c FREE from %d to %d\n", type, start_idx, end_idx);
            }
            else if(n >= 2) {
                // 格式可能是 "N FREE" (整条边界)
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
// 支持 QVAR（时变流量）和 HVAR（时变水位）
bool read_bdy(const char* filename, std::vector<PointBoundary>& point_bcs) {
    std::ifstream file(filename);
    if(!file.is_open()) { printf("Error: Cannot open BDY file %s\n", filename); return false; }

    std::string line;
    std::getline(file, line); // 跳过第一行（标题）

    while(std::getline(file, line)) {
        // 跳过空行
        if(line.empty() || line[0] == '#') continue;

        // 查找点边界名称 (精确匹配)
        for(auto& pb : point_bcs) {
            // 只处理QVAR和HVAR类型
            if(pb.type != BC_QVAR && pb.type != BC_HVAR) continue;

            // 精确匹配：名称后面应该是空格、换行或结束
            if(line.find(pb.name) == 0 &&
               (line.length() == pb.name.length() ||
                line[pb.name.length()] == ' ' ||
                line[pb.name.length()] == '\n' ||
                line[pb.name.length()] == '\r')) {
                // 读取数据点数量
                std::getline(file, line);
                int n;
                sscanf(line.c_str(), "%d", &n);

                // 读取时间-值对
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

// 线性插值获取当前时刻值（用于QVAR/HVAR）
double interpolate_bc_value(const PointBoundary& pb, double t) {
    if(pb.times.empty()) return pb.value;  // 如果没有时间序列，返回固定值
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

// 更新Qold数组（加速模式需要） - 将Qx/Qy转换为单位宽度流量(m²/s)
void UpdateQs_GPU(RealView& Qx, RealView& Qy, RealView& Qxold, RealView& Qyold,
                  int xsz, int ysz, double dx) {
    // Qx单位是m³/s，除以dx得到m²/s（单位宽度流量）
    // 原版LISFLOOD：Qx、Qy、Qxold、Qyold数组大小都是 (xsz+1)×(ysz+1)
    // 索引都是 i + j*(xsz+1)
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
// dt = cfl * dx / sqrt(g * Hmax)
void FloodplainQ_GPU(RealView& H, RealView& DEM, RealView& Qx, RealView& Qy,
    RealView& Qxold, RealView& Qyold,
    int xsz, int ysz, double dx, double dy, double FPn, double DepthThresh,
    double MaxHflow, double dhlin, double& Tstep, double Qlimfact, double dA,
    double cfl, double g, double InitTstep, double theta, double nodata) {

    // 1. 使用parallel_reduce找最大水深（只计算有效单元）
    double Hmax = 0.0;
    Kokkos::parallel_reduce("FindHmax", xsz * ysz,
        KOKKOS_LAMBDA(int idx, double& max_h) {
            if(DEM(idx) != nodata && H(idx) > max_h) max_h = H(idx);
        }, Kokkos::Max<double>(Hmax));

    // 2. 根据最大水深计算自适应时间步（LISFLOOD acceleration公式）
    double dt;
    if(Hmax > DepthThresh) {
        dt = cfl * dx / sqrt(g * Hmax);
    } else {
        dt = InitTstep;  // 全干时使用初始时间步
    }

    // 更新Tstep供主循环使用
    Tstep = dt;

    // 计算Qx - 加速模式（半隐式动量方程）
    // 公式来自 fp_acc.cpp:76
    // 重要：原版使用MaskTest检查单元是否有效，这里检查nodata
    Kokkos::parallel_for("Qx", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{xsz-1,ysz}),
        KOKKOS_LAMBDA(int i, int j) {
            int p0 = i + j * xsz, p1 = i + 1 + j * xsz;
            int pq0 = (i+1) + j * (xsz+1);

            double z0 = DEM(p0), z1 = DEM(p1);

            // 检查nodata：如果任一单元是nodata，流量为0
            // 这是原版LISFLOOD的MaskTest功能
            if(z0 == nodata || z1 == nodata) {
                Qx(pq0) = 0.0;
                return;
            }

            double h0 = H(p0), h1 = H(p1);
            double fn = FPn;
            double y0 = z0 + h0, y1 = z1 + h1;
            double dh = y0 - y1;
            double Sf = -dh / dx;  // 水面坡度（注意符号）
            double hflow = getmax(y0, y1) - getmax(z0, z1);
            hflow = getmax(hflow, 0.0);
            hflow = getmin(hflow, MaxHflow);

            double Q = 0.0;

            // 加速模式：只有当hflow > 阈值时才计算流量
            if(hflow > DepthThresh) {
                // 读取前一时间步的流量（单位：m²/s）
                double q0 = Qxold(pq0);

                // 获取相邻单元的前一步流量（用于q-centred scheme）
                double qup = (i > 0) ? Qxold(pq0-1) : 0.0;
                double qdown = (i < xsz-2) ? Qxold(pq0+1) : 0.0;

                // 计算2D摩擦项：需要Qy在相邻4个角点的平均值
                // 原版公式（fp_acc.cpp:42-46）：Qyold数组大小为 (xsz+1)×(ysz+1)
                double qy_avg = 0.0;
                int pqy1 = i + j*(xsz+1);           // 左下
                int pqy2 = (i+1) + j*(xsz+1);       // 右下
                int pqy3 = i + (j+1)*(xsz+1);       // 左上
                int pqy4 = (i+1) + (j+1)*(xsz+1);   // 右上
                qy_avg = (Qyold(pqy1) + Qyold(pqy2) + Qyold(pqy3) + Qyold(pqy4)) / 4.0;
                double qvect = sqrt(q0*q0 + qy_avg*qy_avg);

                // q-centred scheme (theta=1.0时退化为半隐式)
                double numerator = (theta*q0 + 0.5*(1.0-theta)*(qup+qdown)) - (g*dt*hflow*Sf);
                double denominator = 1.0 + g*dt*hflow*fn*fn*fabs(qvect) / pow(hflow, 10.0/3.0);
                Q = numerator / denominator * dx;

                // 修正：如果Q和dh方向相反，改用纯半隐式格式（Bates et al 2010）
                if(Q * dh < 0.0) {
                    numerator = q0 - (g*dt*hflow*Sf);
                    Q = numerator / denominator * dx;
                }
            }

            Qx(pq0) = Q;
        });

    // 计算Qy - 加速模式（半隐式动量方程）
    Kokkos::parallel_for("Qy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{xsz,ysz-1}),
        KOKKOS_LAMBDA(int i, int j) {
            int p0 = i + j * xsz, p1 = i + (j+1) * xsz;
            // 原版Qy索引：i + (j+1)*(xsz+1) (fp_acc.cpp:114)
            int pq0 = i + (j+1) * (xsz+1);

            double z0 = DEM(p0), z1 = DEM(p1);

            // 检查nodata：如果任一单元是nodata，流量为0
            if(z0 == nodata || z1 == nodata) {
                Qy(pq0) = 0.0;
                return;
            }

            double h0 = H(p0), h1 = H(p1);
            double fn = FPn;
            double y0 = z0 + h0, y1 = z1 + h1;
            double dh = y0 - y1;
            double Sf = -dh / dx;  // 使用dx（均匀网格）
            double hflow = getmax(y0, y1) - getmax(z0, z1);
            hflow = getmax(hflow, 0.0);
            hflow = getmin(hflow, MaxHflow);

            double Q = 0.0;

            if(hflow > DepthThresh) {
                double q0 = Qyold(pq0);

                // 获取上下相邻的前一步流量 (原版fp_acc.cpp:150-151)
                double qup = (j > 0) ? Qyold(i + j*(xsz+1)) : 0.0;
                double qdown = (j < ysz-2) ? Qyold(i + (j+2)*(xsz+1)) : 0.0;

                // 计算2D摩擦项：Qx在相邻4个角点的平均值
                // 原版公式（fp_acc.cpp:125-130）：Qxold数组大小为 (xsz+1)×(ysz+1)
                double qx_avg = 0.0;
                int pqx1 = i + j*(xsz+1);           // 左下
                int pqx2 = (i+1) + j*(xsz+1);       // 右下
                int pqx3 = i + (j+1)*(xsz+1);       // 左上
                int pqx4 = (i+1) + (j+1)*(xsz+1);   // 右上
                qx_avg = (Qxold(pqx1) + Qxold(pqx2) + Qxold(pqx3) + Qxold(pqx4)) / 4.0;
                double qvect = sqrt(q0*q0 + qx_avg*qx_avg);

                // q-centred scheme
                double numerator = (theta*q0 + 0.5*(1.0-theta)*(qup+qdown)) - (g*dt*hflow*Sf);
                double denominator = 1.0 + g*dt*hflow*fn*fn*fabs(qvect) / pow(hflow, 10.0/3.0);
                Q = numerator / denominator * dx;

                // 修正
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
// 支持闭合边界（默认）和自由出流边界（FREE）
// 原版LISFLOOD: boundary.cpp，每个时间步都调用
// 自由出流边界：水可以根据水面坡度流出域外
void BCs_GPU(RealView& Qx, RealView& Qy, RealView& H, RealView& DEM,
             int xsz, int ysz, double dx, double g, double dt, double nodata,
             bool west_free, bool east_free, bool north_free, bool south_free) {

    // 西边界 (i=0)
    if(west_free) {
        // FREE边界：允许出流，根据曼宁公式计算
        Kokkos::parallel_for("BCs_West_FREE", ysz, KOKKOS_LAMBDA(int j) {
            int p0 = 0 + j * xsz;  // 边界单元
            double h0 = H(p0);
            double z0 = DEM(p0);
            if(z0 == nodata || h0 < 0.001) {
                Qx(j * (xsz+1)) = 0.0;
            } else {
                // 水面高程坡度（假设外部水面低）
                // 简化处理：用相邻单元的坡度
                int p1 = 1 + j * xsz;
                double h1 = H(p1);
                double z1 = DEM(p1);
                double y0 = z0 + h0;
                double y1 = z1 + h1;
                // 如果边界单元水位更高，允许出流
                if(y0 > z0) {
                    double Sf = (y1 - y0) / dx;  // 向外的坡度
                    if(Sf < 0) {
                        // 水向外流，用简单公式 Q = h * sqrt(g*h) * dx
                        double Q = -h0 * sqrt(g * h0) * dx * 0.5;  // 负号表示向西流出
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
        // CLOSED边界
        Kokkos::parallel_for("BCs_West_CLOSED", ysz + 1, KOKKOS_LAMBDA(int j) {
            Qx(j * (xsz+1)) = 0.0;
        });
    }

    // 东边界 (i=xsz)
    if(east_free) {
        Kokkos::parallel_for("BCs_East_FREE", ysz, KOKKOS_LAMBDA(int j) {
            int p0 = (xsz-1) + j * xsz;  // 边界单元
            double h0 = H(p0);
            double z0 = DEM(p0);
            if(z0 == nodata || h0 < 0.001) {
                Qx(xsz + j * (xsz+1)) = 0.0;
            } else {
                // 如果有水，允许出流
                if(h0 > 0.001) {
                    double Q = h0 * sqrt(g * h0) * dx * 0.5;  // 正号表示向东流出
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
            int p0 = i + 0 * xsz;  // 边界单元
            double h0 = H(p0);
            double z0 = DEM(p0);
            if(z0 == nodata || h0 < 0.001) {
                Qy(i) = 0.0;
            } else {
                if(h0 > 0.001) {
                    double Q = -h0 * sqrt(g * h0) * dx * 0.5;  // 负号表示向北流出
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
            int p0 = i + (ysz-1) * xsz;  // 边界单元
            double h0 = H(p0);
            double z0 = DEM(p0);
            if(z0 == nodata || h0 < 0.001) {
                Qy(i + ysz * (xsz+1)) = 0.0;
            } else {
                if(h0 > 0.001) {
                    double Q = h0 * sqrt(g * h0) * dx * 0.5;  // 正号表示向南流出
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
// 在每个时间步后应用点边界条件
void ApplyPointBCs_GPU(RealView& H, IntView& bc_idx, IntView& bc_type, RealView& bc_value,
                       int num_bcs, int ncols, double dx, double dA, double Tstep) {
    // bc_type: 1=FREE, 2=HFIX, 3=HVAR（水位类型），4=QFIX, 5=QVAR（流量类型）
    // 注意：流量边界在AddSources阶段处理，这里主要处理水位边界和FREE边界

    Kokkos::parallel_for("ApplyPointBCs", num_bcs, KOKKOS_LAMBDA(int b) {
        int idx = bc_idx(b);
        int type = bc_type(b);
        double val = bc_value(b);

        if(type == 1) {
            // BC_FREE: 排水口，水深强制为0
            H(idx) = 0.0;
        }
        else if(type == 2 || type == 3) {
            // BC_HFIX / BC_HVAR: 固定/时变水位
            // 强制水深为指定值
            H(idx) = val;
        }
        // 流量边界(QFIX/QVAR)在AddSources阶段处理，这里不处理
    });

    Kokkos::fence();
}

// GPU水深更新
void UpdateH_GPU(RealView& H, RealView& Qx, RealView& Qy, int xsz, int ysz, double dA, double Tstep) {
    // 原版LISFLOOD (iterateq.cpp:100)：
    // dV = Tstep * (Qx[i+j*(xsz+1)] - Qx[i+1+j*(xsz+1)] + Qy[i+j*(xsz+1)] - Qy[i+(j+1)*(xsz+1)])
    Kokkos::parallel_for("UpdateH", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{xsz,ysz}),
        KOKKOS_LAMBDA(int i, int j) {
            int idx = i + j * xsz;
            // Qx和Qy都是 (xsz+1) x (ysz+1)，索引都是 i + j*(xsz+1)
            double Qx_in = Qx(i + j*(xsz+1)), Qx_out = Qx((i+1) + j*(xsz+1));
            double Qy_in = Qy(i + j*(xsz+1)), Qy_out = Qy(i + (j+1)*(xsz+1));
            H(idx) += Tstep * (Qx_in - Qx_out + Qy_in - Qy_out) / dA;
            if(H(idx) < 0.0) H(idx) = 0.0;
        });
    Kokkos::fence();
}

// GPU流速计算 - 从Qx/Qy计算单元中心的流速
// 公式：V = Q / (dx * hflow)，参考fp_acc.cpp:93
void ComputeVelocity_GPU(RealView& Vx_out, RealView& Vy_out,
                         RealView& H, RealView& DEM, RealView& Qx, RealView& Qy,
                         int xsz, int ysz, double dx, double nodata) {
    // 计算单元中心的X方向流速（取左右边界的平均）
    Kokkos::parallel_for("ComputeVx", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{xsz,ysz}),
        KOKKOS_LAMBDA(int i, int j) {
            int idx = i + j * xsz;
            double z = DEM(idx);
            double h = H(idx);

            if(z == nodata || h < 0.001) {
                Vx_out(idx) = 0.0;
                return;
            }

            // 左边界和右边界的Qx
            double Qx_left = Qx(i + j*(xsz+1));
            double Qx_right = Qx((i+1) + j*(xsz+1));

            // 流动水深（简化：使用当前单元水深）
            double hflow = h;
            if(hflow < 0.001) hflow = 0.001;

            // 计算边界流速并取平均
            double Vx_left = (fabs(Qx_left) > 1e-10) ? Qx_left / dx / hflow : 0.0;
            double Vx_right = (fabs(Qx_right) > 1e-10) ? Qx_right / dx / hflow : 0.0;

            // 单元中心流速：取绝对值较大的那个（保持方向）
            if(fabs(Vx_left) > fabs(Vx_right)) {
                Vx_out(idx) = Vx_left;
            } else {
                Vx_out(idx) = Vx_right;
            }
        });

    // 计算单元中心的Y方向流速
    Kokkos::parallel_for("ComputeVy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{xsz,ysz}),
        KOKKOS_LAMBDA(int i, int j) {
            int idx = i + j * xsz;
            double z = DEM(idx);
            double h = H(idx);

            if(z == nodata || h < 0.001) {
                Vy_out(idx) = 0.0;
                return;
            }

            // 上边界和下边界的Qy
            double Qy_top = Qy(i + j*(xsz+1));
            double Qy_bottom = Qy(i + (j+1)*(xsz+1));

            // 流动水深
            double hflow = h;
            if(hflow < 0.001) hflow = 0.001;

            // 计算边界流速
            double Vy_top = (fabs(Qy_top) > 1e-10) ? Qy_top / dx / hflow : 0.0;
            double Vy_bottom = (fabs(Qy_bottom) > 1e-10) ? Qy_bottom / dx / hflow : 0.0;

            // 单元中心流速
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

    double rain_depth = rain_rate * Tstep;  // 降雨深度 (m)

    Kokkos::parallel_for("Rainfall", xsz * ysz, KOKKOS_LAMBDA(int idx) {
        if(DEM(idx) != nodata) {
            H(idx) += rain_depth;
            if(H(idx) < 0.0) H(idx) = 0.0;  // 确保非负（处理蒸发情况）
        }
    });
    Kokkos::fence();
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

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        printf("=== LISFLOOD-GPU Point Source Simulation ===\n");
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

        // 读取边界条件（使用LISFLOOD原版坐标系统）
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

        // 分离流量边界（QVAR/QFIX）和其他点边界
        std::vector<PointBoundary> flow_bcs;  // QVAR/QFIX
        std::vector<PointBoundary> other_bcs; // FREE/HFIX/HVAR
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
                // 创建stage输出文件
                char stage_filename[256];
                sprintf(stage_filename, "%s/stage.csv", par.output_dir.c_str());
                stage_fp = fopen(stage_filename, "w");
                if(stage_fp) {
                    // 写入头部
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
        double g = 9.81;  // 重力加速度 (m/s²)
        double theta = 1.0;  // 加速模式参数 (1.0=半隐式，默认值)
        double sim_time = par.sim_time;
        double save_int = par.save_int;
        double mass_int = par.mass_int;

        // 创建输出目录
        mkdir(par.output_dir.c_str(), 0755);

        // 分配GPU数组
        RealView H("H", ncols * nrows);
        RealView DEM("DEM", ncols * nrows);
        // 重要：原版LISFLOOD中Qx和Qy数组大小都是 (xsz+1)×(ysz+1)！
        // 见 iterateq.cpp:80-82
        RealView Qx("Qx", (ncols+1) * (nrows+1));
        RealView Qy("Qy", (ncols+1) * (nrows+1));
        // 加速模式需要存储前一时间步的流量（单位：m²/s）
        RealView Qxold("Qxold", (ncols+1) * (nrows+1));
        RealView Qyold("Qyold", (ncols+1) * (nrows+1));

        // 流速数组（可选，用于输出）
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

        // 初始化H=0, Qx=0, Qy=0, Qxold=0, Qyold=0
        Kokkos::parallel_for("InitH", ncols*nrows, KOKKOS_LAMBDA(int i) { H(i) = 0.0; });
        Kokkos::parallel_for("InitQx", (ncols+1)*(nrows+1), KOKKOS_LAMBDA(int i) { Qx(i) = 0.0; });
        Kokkos::parallel_for("InitQy", (ncols+1)*(nrows+1), KOKKOS_LAMBDA(int i) { Qy(i) = 0.0; });
        Kokkos::parallel_for("InitQxold", (ncols+1)*(nrows+1), KOKKOS_LAMBDA(int i) { Qxold(i) = 0.0; });
        Kokkos::parallel_for("InitQyold", (ncols+1)*(nrows+1), KOKKOS_LAMBDA(int i) { Qyold(i) = 0.0; });
        Kokkos::fence();

        printf("\nStarting simulation: %.0f seconds, output every %.0f seconds\n\n", sim_time, save_int);
        fflush(stdout);

        // 预分配流量边界索引数组 (GPU) - QVAR/QFIX
        int num_flow_bcs = flow_bcs.size();
        IntView ps_idx("ps_idx", num_flow_bcs > 0 ? num_flow_bcs : 1);
        RealView ps_flow("ps_flow", num_flow_bcs > 0 ? num_flow_bcs : 1);
        auto ps_idx_host = Kokkos::create_mirror_view(ps_idx);
        auto ps_flow_host = Kokkos::create_mirror_view(ps_flow);
        for(int s = 0; s < num_flow_bcs; s++) {
            ps_idx_host(s) = flow_bcs[s].idx;
        }
        Kokkos::deep_copy(ps_idx, ps_idx_host);

        // 预分配其他点边界数组 (GPU) - FREE/HFIX/HVAR
        int num_other_bcs = other_bcs.size();
        IntView bc_idx("bc_idx", num_other_bcs > 0 ? num_other_bcs : 1);
        IntView bc_type("bc_type", num_other_bcs > 0 ? num_other_bcs : 1);
        RealView bc_value("bc_value", num_other_bcs > 0 ? num_other_bcs : 1);
        auto bc_idx_host = Kokkos::create_mirror_view(bc_idx);
        auto bc_type_host = Kokkos::create_mirror_view(bc_type);
        auto bc_value_host = Kokkos::create_mirror_view(bc_value);
        for(int b = 0; b < num_other_bcs; b++) {
            bc_idx_host(b) = other_bcs[b].idx;
            // 类型映射：FREE=1, HFIX=2, HVAR=3
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
        double next_stage = par.stage_int;  // Stage输出时间
        int save_num = 0;

        // 记录开始时间（wall clock time）
        auto wall_start = std::chrono::high_resolution_clock::now();

        // 保存初始状态（t=0）- 与CPU版本的out-0000对应
        if(par.async_output && async_output.is_running()) {
            OutputData out_data;
            out_data.time = 0.0;
            out_data.save_num = save_num;
            out_data.H.resize(ncols * nrows, 0.0);  // 初始水深为0
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

            // 1. 计算流量（加速模式：半隐式动量方程）
            FloodplainQ_GPU(H, DEM, Qx, Qy, Qxold, Qyold, ncols, nrows, dx, dy, FPn,
                           DepthThresh, MaxHflow, dhlin, Tstep, Qlimfact, dA,
                           cfl, g, InitTstep, theta, nodata);

            // 2. 边界条件处理
            // 支持闭合边界（默认）和自由出流边界（FREE）
            BCs_GPU(Qx, Qy, H, DEM, ncols, nrows, dx, g, Tstep, nodata,
                    west_free, east_free, north_free, south_free);

            // 前10步每步输出，之后每100步输出进度
            if(step < 10 || step % 100 == 0) {
                printf("Step %d, t=%.2fs, dt=%.4fs\n", step, t, Tstep);
                fflush(stdout);
            }

            // 3. 添加流量边界 (QVAR/QFIX)
            // LISFLOOD原版：Q是单位宽度流量(m²/s)，需乘dx得到体积流量
            // 公式：H += q * dx * dt / dA（iterateq.cpp:71）
            if(num_flow_bcs > 0) {
                for(int s = 0; s < num_flow_bcs; s++) {
                    double flow_val;
                    if(flow_bcs[s].type == BC_QFIX) {
                        // 固定流量
                        flow_val = flow_bcs[s].value;
                    } else {
                        // 时变流量
                        flow_val = interpolate_bc_value(flow_bcs[s], t);
                    }
                    ps_flow_host(s) = flow_val * dx * Tstep / dA;
                }
                Kokkos::deep_copy(ps_flow, ps_flow_host);
                Kokkos::parallel_for("AddFlowBCs", num_flow_bcs, KOKKOS_LAMBDA(int s) {
                    int idx = ps_idx(s);
                    H(idx) += ps_flow(s);
                    // 负流量保护：防止水深变为负值（双向耦合时重要）
                    if(H(idx) < 0.0) H(idx) = 0.0;
                });
                Kokkos::fence();
            }

            // 4. 添加降雨（如果启用）
            if(rainfall.enabled) {
                double rain_rate = interpolate_rain(rainfall, t);
                Rainfall_GPU(H, DEM, ncols, nrows, rain_rate, Tstep, nodata);
            }

            // 5. 更新水深
            UpdateH_GPU(H, Qx, Qy, ncols, nrows, dA, Tstep);

            // 6. 应用点边界条件 (FREE/HFIX/HVAR)
            if(num_other_bcs > 0) {
                // 更新HVAR边界值
                for(int b = 0; b < num_other_bcs; b++) {
                    if(other_bcs[b].type == BC_HVAR) {
                        bc_value_host(b) = interpolate_bc_value(other_bcs[b], t);
                    }
                }
                Kokkos::deep_copy(bc_value, bc_value_host);
                ApplyPointBCs_GPU(H, bc_idx, bc_type, bc_value, num_other_bcs, ncols, dx, dA, Tstep);
            }

            // 7. 更新Qold数组（加速模式需要）
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
                    // 异步输出：复制数据到队列
                    OutputData out_data;
                    out_data.time = t;
                    out_data.save_num = save_num;

                    // 复制水深数据
                    auto H_host = Kokkos::create_mirror_view(H);
                    Kokkos::deep_copy(H_host, H);
                    out_data.H.resize(ncols * nrows);
                    for(int i = 0; i < ncols * nrows; i++) {
                        out_data.H[i] = H_host(i);
                    }

                    // 如果启用流速输出，计算并复制流速
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
                    // 同步输出（原有方式）
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

        // 记录结束时间
        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_seconds = std::chrono::duration<double>(wall_end - wall_start).count();

        // 等待异步输出完成
        if(par.async_output && async_output.is_running()) {
            printf("\nWaiting for async output to complete...\n");
            async_output.finish();
            printf("Async output finished.\n");
        }

        // 保存最终状态 H_final.asc（始终同步保存）
        char final_file[256];
        sprintf(final_file, "%s/H_final.asc", par.output_dir.c_str());
        write_asc(final_file, H, ncols, nrows, xll, yll, cellsize);
        printf("Saved: %s\n", final_file);

        // 如果启用流速输出，也保存最终流速
        if(par.voutput) {
            ComputeVelocity_GPU(Vx, Vy, H, DEM, Qx, Qy, ncols, nrows, dx, nodata);
            sprintf(final_file, "%s/Vx_final.asc", par.output_dir.c_str());
            write_asc(final_file, Vx, ncols, nrows, xll, yll, cellsize);
            printf("Saved: %s\n", final_file);
            sprintf(final_file, "%s/Vy_final.asc", par.output_dir.c_str());
            write_asc(final_file, Vy, ncols, nrows, xll, yll, cellsize);
            printf("Saved: %s\n", final_file);
        }

        // 关闭Stage输出文件
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
