# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 2.8.7)
set(PROJECT_NAME pointpillars)
# 执行系统命令`uname -m`来检测当前系统的架构，并将输出结果存储到变量ARCH中
EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCH )
message( STATUS "Architecture: ${ARCH}" )
# 寻找CUDA包，这是编译.cuda源文件所必需的
find_package(CUDA REQUIRED)
# 指定CUDA的版本号和安装路径
set(CUDA_VERSION 11.3)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-${CUDA_VERSION})
# 设置构建类型为Release模式，以优化性能
SET(CMAKE_BUILD_TYPE "Release")
# 添加编译器警告标志和C++11标准支持
add_compile_options(-W)
add_compile_options(-std=c++11)
# 枚举所有可能的CUDA设备架构，并为它们设置适当的NVCC编译器标志
set(SMS 50 52 53 60 61 62 70 72 75 80 86)
foreach(sm ${SMS})
    set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
# 获取支持的最新的CUDA设备架构，并为其添加额外的编译标志
list(GET SMS -1 LATEST_SM)
set(GENCODE "${GENCODE} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
# 设置CUDA的NVCC编译器标志，包括交叉编译器、包含目录和链接器标志
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -ccbin ${CMAKE_CXX_COMPILER}
    -Xcompiler -DWIN_INTERFACE_CUSTOM
    -Xcompiler -I/usr/${ARCH}-linux-gnu/include/
    -Xlinker -lsocket
    -Xlinker -rpath=/usr/lib/${ARCH}-linux-gnu/
    -Xlinker -rpath=/usr/${ARCH}-linux-gnu/lib/
    -Xlinker -L/usr/lib/${ARCH}-linux-gnu/
    -Xlinker -L/usr/${ARCH}-linux-gnu/lib/
)
# 设置TensorRT的包含目录和库目录的路径
set(TENSORRT_INCLUDE_DIRS /usr/include/${ARCH}-linux-gnu/)
set(TENSORRT_LIBRARY_DIRS /usr/lib/${ARCH}-linux-gnu/)
# 将CUDA和TensorRT的包含目录，以及项目的include目录加入到包含目录中
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    ../include/
)
# 将TensorRT的库目录和系统的库目录加入到链接目录中
link_directories(
    ${TENSORRT_LIBRARY_DIRS}
    /usr/lib/${ARCH}-linux-gnu
    /usr/${ARCH}-linux-gnu/lib/
)
# 递归地将项目src目录下的所有.cu和.cpp文件加入到源文件列表
file(GLOB_RECURSE SOURCE_FILES
    ../src/*.cu
    ../src/*.cpp
)
# 使用CUDA_ADD_EXECUTABLE来添加一个可执行文件，它会自动处理CUDA文件的编译
cuda_add_executable(${PROJECT_NAME} main.cpp ${SOURCE_FILES})
# 将目标（pointpillars）与TensorRT库进行链接，确保可执行文件可以调用TensorRT的功能
target_link_libraries(${PROJECT_NAME}
    libnvinfer.so
    libnvonnxparser.so
    libnvinfer_plugin.so
)
