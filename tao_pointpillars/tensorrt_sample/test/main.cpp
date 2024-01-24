/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cassert>
#include <iostream>
#include <sstream>
#include <fstream>
#include <unistd.h>
#include <string>
#include "cuda_runtime.h"
#include "./pointpillar.h"

#include <boost/filesystem/convenience.hpp>

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}
// 加载二进制文件到内存
/*
这样通过二进制方式打开文件,一次性读取整个文件,避免多次小范围读取文件内容。
并将文件内容加载到一段连续的内存buffer中,方便后续直接访问处理。

data和length作为输出参数,返回加载的内存buffer指针和长度,供调用者使用。

这是一个常见的加载二进制文件到内存的实现方法。这里加载的是点云数据二进制文件,后面可以直接在内存中对点云数据进行处理。
*/
int loadData(const char *file, void **data, unsigned int *length)
{
  //打开二进制文件文件流dataFile,读取模式。
  std::fstream dataFile(file, std::ifstream::in);

  if (!dataFile.is_open())
  {
    std::cout << "Can't open files: "<< file<<std::endl;
    return -1;
  }

  //get length of file:获取文件长度len:先定位到文件末尾,tellg()获取当前位置就是文件长度;然后重新定位到文件开头。
  unsigned int len = 0;
  dataFile.seekg (0, dataFile.end);
  len = dataFile.tellg();
  dataFile.seekg (0, dataFile.beg);

  //allocate memory:分配大小为len的buffer。
  char *buffer = new char[len];
  if(buffer==NULL) {
    std::cout << "Can't malloc buffer."<<std::endl;
    dataFile.close();
    exit(-1);
  }

  //read data as a block: 一次性读取整个文件到buffer中,read函数将文件内容读取到buffer。
  dataFile.read(buffer, len);
  dataFile.close();
//   将buffer的地址和长度赋值给输出参数data和length。
  *data = (void*)buffer;
  *length = len;
  return 0;  
}

void split_str(
    const char* s,
    std::vector<std::string>& ret,  // NOLINT(runtime/references)
    char del = ',') {
    int idx = 0;
    auto p = std::string(s + idx).find(std::string(1, del));
    while (std::string::npos != p) {
        auto s_tmp = std::string(s + idx).substr(0, p);
        ret.push_back(s_tmp);
        idx += (p + 1);
        p = std::string(s + idx).find(std::string(1, del));
    }
    if (s[idx] != 0) {
        ret.push_back(std::string(s + idx));
    }
}
//解析命令行参数,获取模型、数据、输出等路径参数
void parse_args(
  int argc, char**argv,
  std::vector<std::string>& class_names,
  float& nms_iou_thresh,
  int& pre_nms_top_n,
  bool& do_profile,
  std::string& model_path,
  std::string& engine_path,
  std::string& data_path,
  std::string& data_type,
  std::string& output_path
  ) {
    int c;
    while ((c = getopt(argc, argv, "c:n:t:m:l:d:e:o:ph")) != -1) {
        switch (c) {
            case 't':
                {
                    nms_iou_thresh = atof(optarg);
                    break;
                }
            case 'n':
                {
                    pre_nms_top_n = atoi(optarg);
                    break;
                }
            case 'c':
                {
                    split_str(optarg, class_names);
                    break;
                }
            case 'm':
                {
                    model_path = std::string(optarg);
                    break;
                }
            case 'e':
                {
                    engine_path = std::string(optarg);
                    break;
                }
            case 'l':
                {
                    data_path = std::string(optarg);
                    break;
                }
            case 'o':
                {
                    output_path = std::string(optarg);
                    break;
                }
            case 'd':
                {
                    data_type = std::string(optarg);
                    break;
                }
            case 'p':
                {
                    do_profile = true;
                    break;
                }
            case 'h':
                {
                  std::cout << "Usage: " << std::endl;
                  std::cout << argv[0] << " -t <nms_iou_thresh>" <<
                   " -c <class_names> -n <pre_nms_top_n>" <<
                   " -l <LIDAR_data_path> -m <model_path>" <<
                   " -e <engine_path> -d <data_type> -o <output_path> -p -h" <<
                   std::endl;
                  exit(1);
                }
            default:
                {
                    std::cerr << "Unrecognized argument" << std::endl;
                    abort();
                }
        }
    }
}

std::vector<std::string> class_names;
float nms_iou_thresh;
int pre_nms_top_n;
bool do_profile{false};
std::string model_path;
std::string engine_path;
std::string data_path;
std::string data_type{"fp32"};
std::string output_path;

//调用SaveBoxPred保存结果到文件
void SaveBoxPred(std::vector<Bndbox> boxes, std::string file_name)
{
    std::ofstream ofs;
    ofs.open(file_name, std::ios::out);
    if (ofs.is_open()) {
        for (const auto box : boxes) {
          ofs << box.x << " ";
          ofs << box.y << " ";
          ofs << box.z << " ";
          ofs << box.w << " ";
          ofs << box.l << " ";
          ofs << box.h << " ";
          ofs << box.rt << " ";
          ofs << box.id << " ";
          ofs << box.score << " ";
          ofs << "\n";
        }
    }
    else {
      std::cerr << "Output file cannot be opened!" << std::endl;
    }
    ofs.close();
    std::cout << "Saved prediction in: " << file_name << std::endl;
    return;
};


int main(int argc, char **argv)
{
  parse_args(
    argc, argv,
    class_names,
    nms_iou_thresh,
    pre_nms_top_n,
    do_profile,
    model_path,
    engine_path,
    data_path,
    data_type,
    output_path
  );
  assert(data_type == "fp32" || data_type == "fp16");
  std::cout << "Loading Data: " << data_path << std::endl;
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  cudaStream_t stream = NULL;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaStreamCreate(&stream));

  std::vector<Bndbox> nms_pred;
  nms_pred.reserve(100);
    // 创建PointPillar模型实例进行推理
  PointPillar pointpillar(model_path, engine_path, stream, data_type);

  
    std::string dataFile = data_path;
    //load points cloud
    unsigned int length = 0;
    void *data = NULL;
    std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
    loadData(dataFile.data(), &data, &length);
    buffer.reset((char *)data);

    float* points = (float*)buffer.get();
    unsigned int num_point_values = pointpillar.getPointSize();
    unsigned int points_size = length/sizeof(float)/num_point_values;

    float *points_data = nullptr;
    unsigned int *points_num = nullptr;
    unsigned int points_data_size = points_size * num_point_values * sizeof(float);
    
    checkCudaErrors(cudaMallocManaged((void **)&points_data, points_data_size));
    checkCudaErrors(cudaMallocManaged((void **)&points_num, sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(points_data, points, points_data_size, cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(points_num, &points_size, sizeof(unsigned int), cudaMemcpyDefault));
    checkCudaErrors(cudaDeviceSynchronize());

    cudaEventRecord(start, stream);

    pointpillar.doinfer(
      points_data, points_num, nms_pred,
      nms_iou_thresh,
      pre_nms_top_n,
      class_names,
      do_profile
    );
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout<<"TIME: pointpillar: "<< elapsedTime <<" ms." <<std::endl;

    checkCudaErrors(cudaFree(points_data));
    checkCudaErrors(cudaFree(points_num));
    std::cout<<"Bndbox objs: "<< nms_pred.size()<<std::endl;
    
    
    std::string bin_file_name = data_path.substr(0, data_path.find_last_of('.'));
    std::string save_file_name = output_path + bin_file_name.substr(bin_file_name.find_last_of('/') + 1) + ".txt";

    SaveBoxPred(nms_pred, save_file_name);
    nms_pred.clear();
    std::cout << ">>>>>>>>>>>" <<std::endl;
  

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaStreamDestroy(stream));

  return 0;
  }

