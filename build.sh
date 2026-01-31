#!/bin/bash
# LISFLOOD-GPU 编译脚本

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

echo "=== LISFLOOD-GPU 编译 ==="
echo "项目目录: $PROJECT_DIR"
echo "编译目录: $BUILD_DIR"
echo ""

# 创建build目录
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# CMake配置
echo "CMake 配置..."
cmake .. -DKokkos_DIR=/usr/local/lib/cmake/Kokkos

if [ $? -ne 0 ]; then
    echo "CMake配置失败!"
    exit 1
fi

# 编译
echo ""
echo "开始编译..."
make -j4

if [ $? -eq 0 ]; then
    echo ""
    echo "编译成功!"
    echo "可执行文件: $BUILD_DIR/lisflood_gpu"
    echo ""
    echo "运行测试:"
    echo "  cd $PROJECT_DIR/test/test01"
    echo "  bash run_test.sh"
else
    echo "编译失败!"
    exit 1
fi
