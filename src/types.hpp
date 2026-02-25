// types.hpp - 公共类型定义
#ifndef LISFLOOD_TYPES_HPP
#define LISFLOOD_TYPES_HPP

#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

// Kokkos View 类型定义
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

// 降雨时间序列结构
struct RainfallData {
    std::vector<double> times;     // 时间点
    std::vector<double> rates;     // 降雨率 (m/s)
    bool enabled = false;
};

#endif // LISFLOOD_TYPES_HPP
