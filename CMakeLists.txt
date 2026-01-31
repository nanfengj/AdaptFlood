cmake_minimum_required(VERSION 3.16)
project(lisflood_gpu_2026 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 Kokkos
find_package(Kokkos REQUIRED)

# 查找 NetCDF（可选，用于nc格式输出）
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(NETCDF QUIET netcdf)
endif()

# 如果pkg-config没找到，手动查找
if(NOT NETCDF_FOUND)
    find_path(NETCDF_INCLUDE_DIRS netcdf.h HINTS /usr/include /usr/local/include)
    find_library(NETCDF_LIBRARIES netcdf HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
    if(NETCDF_INCLUDE_DIRS AND NETCDF_LIBRARIES)
        set(NETCDF_FOUND TRUE)
    endif()
endif()

if(NETCDF_FOUND)
    message(STATUS "NetCDF found: ${NETCDF_LIBRARIES}")
    add_definitions(-DUSE_NETCDF)
else()
    message(STATUS "NetCDF not found, will use ASC output only")
endif()

# ============================================
# 主程序: lisflood_gpu (模块化版本)
# 完整的GPU洪水模拟程序，读取dem/bci/bdy文件
# ============================================
set(LISFLOOD_SOURCES
    src/main.cpp
    src/io.cpp
    src/kernels.cpp
    src/async_output.cpp
)

add_executable(lisflood_gpu ${LISFLOOD_SOURCES})
target_include_directories(lisflood_gpu PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(lisflood_gpu Kokkos::kokkos)

# 链接NetCDF（如果找到）
if(NETCDF_FOUND)
    target_include_directories(lisflood_gpu PRIVATE ${NETCDF_INCLUDE_DIRS})
    target_link_libraries(lisflood_gpu ${NETCDF_LIBRARIES})
endif()

# 链接pthread（用于异步输出）
find_package(Threads REQUIRED)
target_link_libraries(lisflood_gpu Threads::Threads)


# ============================================
# GPU核心函数库 (参考用)
# ============================================
add_library(lisflood_gpu_core STATIC lisflood_gpu_core.cpp)
target_link_libraries(lisflood_gpu_core Kokkos::kokkos)

# ============================================
# 测试程序: test_point_source
# 50x50小网格测试，验证GPU函数
# ============================================
add_executable(test_point_source test_point_source.cpp)
target_link_libraries(test_point_source lisflood_gpu_core Kokkos::kokkos)

# ============================================
# 环境测试: test_kokkos_cuda
# 验证Kokkos+CUDA是否工作
# ============================================
add_executable(test_kokkos_cuda test_kokkos_cuda.cpp)
target_link_libraries(test_kokkos_cuda Kokkos::kokkos)

# 输出编译信息
message(STATUS "=== LISFLOOD-GPU-2026 ===")
message(STATUS "Kokkos devices: ${Kokkos_DEVICES}")
message(STATUS "Kokkos dir: ${Kokkos_DIR}")
