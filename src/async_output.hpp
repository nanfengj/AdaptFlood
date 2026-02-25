// async_output.hpp - 异步输出管理器
#ifndef LISFLOOD_ASYNC_OUTPUT_HPP
#define LISFLOOD_ASYNC_OUTPUT_HPP

#include "types.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <string>

#ifdef USE_NETCDF
#include <netcdf.h>
#define NC_CHECK(e) { int _nc_err = (e); if(_nc_err != NC_NOERR) { \
    printf("NetCDF error: %s\n", nc_strerror(_nc_err)); } }
#endif

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

    void writer_loop();
    void write_output(const OutputData& data);
    void write_asc_file(const char* filename, const std::vector<double>& data);

public:
    void init(int _ncols, int _nrows, double _xll, double _yll,
              double _cellsize, double _nodata, const std::string& _output_dir,
              int _output_format, bool _voutput);

    void enqueue(const OutputData& data);
    void finish();
    bool is_running() const { return running; }
};

#endif // LISFLOOD_ASYNC_OUTPUT_HPP
