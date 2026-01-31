// async_output.cpp - 异步输出管理器实现
#include "async_output.hpp"
#include <cstdio>

void AsyncOutputManager::init(int _ncols, int _nrows, double _xll, double _yll,
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
        std::string nc_file = output_dir + "/output.nc";
        NC_CHECK(nc_create(nc_file.c_str(), NC_CLOBBER | NC_NETCDF4, &ncid));

        NC_CHECK(nc_def_dim(ncid, "time", NC_UNLIMITED, &time_dimid));
        NC_CHECK(nc_def_dim(ncid, "x", ncols, &x_dimid));
        NC_CHECK(nc_def_dim(ncid, "y", nrows, &y_dimid));

        NC_CHECK(nc_def_var(ncid, "time", NC_DOUBLE, 1, &time_dimid, &time_varid));
        nc_put_att_text(ncid, time_varid, "units", 7, "seconds");

        int dims3d[3] = {time_dimid, y_dimid, x_dimid};
        NC_CHECK(nc_def_var(ncid, "H", NC_FLOAT, 3, dims3d, &H_varid));
        nc_put_att_text(ncid, H_varid, "units", 1, "m");
        nc_put_att_text(ncid, H_varid, "long_name", 11, "water_depth");
        float fill_val = (float)nodata_val;
        nc_put_att_float(ncid, H_varid, "_FillValue", NC_FLOAT, 1, &fill_val);

        if(voutput) {
            NC_CHECK(nc_def_var(ncid, "Vx", NC_FLOAT, 3, dims3d, &Vx_varid));
            nc_put_att_text(ncid, Vx_varid, "units", 3, "m/s");
            nc_put_att_text(ncid, Vx_varid, "long_name", 16, "x_direction_velocity");

            NC_CHECK(nc_def_var(ncid, "Vy", NC_FLOAT, 3, dims3d, &Vy_varid));
            nc_put_att_text(ncid, Vy_varid, "units", 3, "m/s");
            nc_put_att_text(ncid, Vy_varid, "long_name", 16, "y_direction_velocity");
        }

        nc_put_att_double(ncid, NC_GLOBAL, "xllcorner", NC_DOUBLE, 1, &xll);
        nc_put_att_double(ncid, NC_GLOBAL, "yllcorner", NC_DOUBLE, 1, &yll);
        nc_put_att_double(ncid, NC_GLOBAL, "cellsize", NC_DOUBLE, 1, &cellsize);

        NC_CHECK(nc_enddef(ncid));
        printf("NetCDF file created: %s\n", nc_file.c_str());
    }
#endif

    stop_flag = false;
    running = true;
    writer_thread = std::thread(&AsyncOutputManager::writer_loop, this);
}

void AsyncOutputManager::enqueue(const OutputData& data) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        output_queue.push(data);
    }
    cv.notify_one();
}

void AsyncOutputManager::finish() {
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

void AsyncOutputManager::writer_loop() {
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

        write_output(data);
    }

    while(!output_queue.empty()) {
        OutputData data = std::move(output_queue.front());
        output_queue.pop();
        write_output(data);
    }
}

void AsyncOutputManager::write_output(const OutputData& data) {
#ifdef USE_NETCDF
    if(output_format >= 1 && ncid >= 0) {
        printf("[AsyncIO] Writing t=%.0fs to NetCDF (time_index=%zu)...\n", data.time, time_index);
        fflush(stdout);

        size_t start1[1] = {time_index};
        size_t count1[1] = {1};
        NC_CHECK(nc_put_vara_double(ncid, time_varid, start1, count1, &data.time));

        std::vector<float> H_float(data.H.size());
        for(size_t i = 0; i < data.H.size(); i++) {
            H_float[i] = (float)data.H[i];
        }
        size_t start3[3] = {time_index, 0, 0};
        size_t count3[3] = {1, (size_t)nrows, (size_t)ncols};
        NC_CHECK(nc_put_vara_float(ncid, H_varid, start3, count3, H_float.data()));

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

        nc_sync(ncid);
        printf("[AsyncIO] Written t=%.0fs to NetCDF, synced.\n", data.time);
        fflush(stdout);
        time_index++;
    }
#endif

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

void AsyncOutputManager::write_asc_file(const char* filename, const std::vector<double>& data) {
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
