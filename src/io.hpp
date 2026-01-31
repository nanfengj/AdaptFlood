// io.hpp - 文件读写函数声明
#ifndef LISFLOOD_IO_HPP
#define LISFLOOD_IO_HPP

#include "types.hpp"
#include <vector>
#include <string>

// 读取.par参数文件
bool read_par(const char* filename, Parameters& p);

// 读取DEM文件
bool read_dem(const char* filename, int& ncols, int& nrows, double& xll, double& yll,
              double& cellsize, double& nodata, std::vector<double>& data);

// 读取bci边界条件文件
bool read_bci(const char* filename, double xll, double yll, int ncols, int nrows,
              double cellsize, std::vector<PointBoundary>& point_bcs,
              std::vector<LineBoundary>& line_bcs);

// 读取bdy时变边界数据文件
bool read_bdy(const char* filename, std::vector<PointBoundary>& point_bcs);

// 读取降雨文件
bool read_rain(const char* filename, RainfallData& rain);

// 读取Stage监测点文件
bool read_stage_file(const char* filename, double xll, double yll, int nrows, double cellsize,
                     std::vector<StagePoint>& stages);

// 线性插值获取当前时刻值
double interpolate_bc_value(const PointBoundary& pb, double t);

// 降雨率插值
double interpolate_rain(const RainfallData& rain, double t);

// 写ASC输出文件
void write_asc(const char* filename, RealView& H, int ncols, int nrows,
               double xll, double yll, double cellsize);

#endif // LISFLOOD_IO_HPP
