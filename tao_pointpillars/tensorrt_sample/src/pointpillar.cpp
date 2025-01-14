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

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include<map>
#include<algorithm>
#include "cuda_runtime.h"
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "NvInferRuntime.h"
#include "NvInferPlugin.h"
#include "pointpillar.h"

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

//这个SimpleProfiler类实现了一个TensorRT的层级性能分析器。主要功能是:
//继承自nvinfer1::IProfiler,这是一个TensorRT的profile接口。

// 使用方法:
// 创建Profiler实例,传入名字。
// 设置到TensorRT执行上下文context中。
// 执行推理后,可以直接打印profiler,获取性能报告。
// 比如可以比较FP32与FP16模式下的层级性能差异。或者比较优化前后模型的层级改进。
struct SimpleProfiler : public nvinfer1::IProfiler
{
    struct Record
    {
        float time{0};
        int count{0};
    };
    // 在reportLayerTime函数中记录每一层的执行时间和调用次数。
    virtual void reportLayerTime(const char* layerName, float ms) noexcept
    {
        // mProfile是一个map,key是层的名字,value是一个Record结构体,包含时间和调用次数。
        mProfile[layerName].count++;
        mProfile[layerName].time += ms;
        if (std::find(mLayerNames.begin(), mLayerNames.end(), layerName) == mLayerNames.end())
        {
            mLayerNames.push_back(layerName);
        }
    }

    SimpleProfiler(const char* name, const std::vector<SimpleProfiler>& srcProfilers = std::vector<SimpleProfiler>())
        : mName(name)
    {
        for (const auto& srcProfiler : srcProfilers)
        {
            for (const auto& rec : srcProfiler.mProfile)
            {
                auto it = mProfile.find(rec.first);
                if (it == mProfile.end())
                {
                    mProfile.insert(rec);
                }
                else
                {
                    it->second.time += rec.second.time;
                    it->second.count += rec.second.count;
                }
            }
        }
    }

    friend std::ostream& operator<<(std::ostream& out, const SimpleProfiler& value)
    {
        out << "========== " << value.mName << " profile ==========" << std::endl;
        float totalTime = 0;
        std::string layerNameStr = "TensorRT layer name";
        int maxLayerNameLength = std::max(static_cast<int>(layerNameStr.size()), 70);
        for (const auto& elem : value.mProfile)
        {
            totalTime += elem.second.time;
            maxLayerNameLength = std::max(maxLayerNameLength, static_cast<int>(elem.first.size()));
        }

        auto old_settings = out.flags();
        auto old_precision = out.precision();
        // Output header
        {
            out << std::setw(maxLayerNameLength) << layerNameStr << " ";
            out << std::setw(12) << "Runtime, "
                << "%"
                << " ";
            out << std::setw(12) << "Invocations"
                << " ";
            out << std::setw(12) << "Runtime, ms" << std::endl;
        }
        for (size_t i = 0; i < value.mLayerNames.size(); i++)
        {
            const std::string layerName = value.mLayerNames[i];
            auto elem = value.mProfile.at(layerName);
            out << std::setw(maxLayerNameLength) << layerName << " ";
            out << std::setw(12) << std::fixed << std::setprecision(1) << (elem.time * 100.0F / totalTime) << "%"
                << " ";
            out << std::setw(12) << elem.count << " ";
            out << std::setw(12) << std::fixed << std::setprecision(2) << elem.time << std::endl;
        }
        out.flags(old_settings);
        out.precision(old_precision);
        out << "========== " << value.mName << " total runtime = " << totalTime << " ms ==========" << std::endl;

        return out;
    }

private:
    std::string mName;
    std::vector<std::string> mLayerNames;
    std::map<std::string, Record> mProfile;
};


TRT::~TRT(void)
{
  context->destroy();
  engine->destroy();
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
}

TRT::TRT(
  std::string modelFile,
  std::string modelCache,
  cudaStream_t stream,
  const std::string& data_type
):stream_(stream)
{
  initLibNvInferPlugins(&gLogger_, "");
//   检查是否已经有缓存的TensorRT engine文件。
// 如果有,则直接反序列化加载该engine文件,更快。
// 如果没有,则从原始ONNX模型文件中加载并创建engine。
  std::fstream trtCache(modelCache, std::ifstream::in);
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  if (!trtCache.is_open())
  {
    std::cout << "Loading Model: " << modelFile << std::endl;
    std::cout << "Building TRT engine from the model."<<std::endl;
    // define builder 创建Builder,Network和Parser。
    auto builder = (nvinfer1::createInferBuilder(gLogger_));

    // define network
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = (builder->createNetworkV2(explicitBatch));

    // define onnxparser
    auto parser = (nvonnxparser::createParser(*network, gLogger_));
    if (!parser->parseFromFile(modelFile.data(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        std::cerr << ": Failed to parse onnx model file, please check the onnx version and trt support op!"
                  << std::endl;
        exit(-1);
    }
    // dynamic shape 创建optimization profile来支持动态shape
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    // define config
    auto networkConfig = builder->createBuilderConfig();
    if(data_type == "fp16") {
        networkConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "Enabled FP16 data type!" << std::endl;
    }
    nvinfer1::Dims dims{};
    dims.nbDims = 3;
    dims.d[0] = 1;
    auto input0_dims = network->getInput(0)->getDimensions();
    dims.d[1] = input0_dims.d[1];
    dims.d[2] = 4;
    profile->setDimensions("points", nvinfer1::OptProfileSelector::kMIN, dims);
    profile->setDimensions("points", nvinfer1::OptProfileSelector::kOPT, dims);
    profile->setDimensions("points", nvinfer1::OptProfileSelector::kMAX, dims);
    dims.nbDims = 1;
    dims.d[0] = 1;
    profile->setDimensions("num_points", nvinfer1::OptProfileSelector::kMIN, dims);
    profile->setDimensions("num_points", nvinfer1::OptProfileSelector::kOPT, dims);
    profile->setDimensions("num_points", nvinfer1::OptProfileSelector::kMAX, dims);
    networkConfig->addOptimizationProfile(profile);
    // set max workspace
    networkConfig->setMaxWorkspaceSize(size_t(1) << 30);
    // 使用buildEngineWithConfig生成优化后的engine。
    engine = (builder->buildEngineWithConfig(*network, *networkConfig));

    if (engine == nullptr)
    {
      std::cerr << ": engine init null!" << std::endl;
      exit(-1);
    }

    // serialize the engine, then close everything down
    auto trtModelStream = (engine->serialize());
    std::string modelCacheSave = modelFile + ".cache";
    std::fstream trtOut(modelCacheSave, std::ifstream::out);
    if (!trtOut.is_open())
    {
       std::cout << "Can't store trt cache.\n";
       exit(-1);
    }
    trtOut.write((char*)trtModelStream->data(), trtModelStream->size());
    trtOut.close();
    trtModelStream->destroy();

    networkConfig->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();
  } else {
    std::cout << "Loading existing TRT Engine: "
              << modelCache
              << std::endl;
    char *data;
    unsigned int length;
    // get length of file:
    trtCache.seekg(0, trtCache.end);
    length = trtCache.tellg();
    trtCache.seekg(0, trtCache.beg);
    data = (char *)malloc(length);
    if (data == NULL ) {
       std::cout << "Can't malloc data.\n";
       exit(-1);
    }
    trtCache.read(data, length);
    // create context
    auto runtime = nvinfer1::createInferRuntime(gLogger_);
    if (runtime == nullptr) {
        std::cerr << ": runtime null!" << std::endl;
        exit(-1);
    }
    engine = (runtime->deserializeCudaEngine(data, length, 0));
    if (engine == nullptr) {
        std::cerr << ": engine null!" << std::endl;
        exit(-1);
    }
    free(data);
    trtCache.close();
  }
  // 创建Execution Context
  context = engine->createExecutionContext();

}

int TRT::doinfer(void**buffers, bool do_profile)
{
  int status;
  SimpleProfiler profiler("perf"); //创建profiler,用于推理性能分析。
  // 如果需要profile,将profiler设置到context中。
  if(do_profile) 
      context->setProfiler(&profiler);
  status = context->enqueueV2(buffers, stream_, &start); // 调用context的enqueueV2函数执行推理。
  if(do_profile)
      std::cout << profiler;
  if (!status)
  {
      return false;
  }
  return true;
}

nvinfer1::Dims TRT::get_binding_shape(int index)
{
  return context->getBindingDimensions(index);
}

int TRT::getPointSize() {
    return context->getBindingDimensions(0).d[2];
}
//PointPillar类构造函数:使用TensorRT加载ONNX模型到engine
PointPillar::PointPillar(
  std::string modelFile,
  std::string engineFile,
  cudaStream_t stream,
  const std::string& data_type
):stream_(stream)
{

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  trt_.reset(new TRT(modelFile, engineFile, stream_, data_type));

  //output of TRT
  box_size = (trt_->get_binding_shape(2).d[1]) * 9 * sizeof(float);
  checkCudaErrors(cudaMallocManaged((void **)&box_output, box_size));
  checkCudaErrors(cudaMallocManaged((void **)&box_num, sizeof(int)));
  res.reserve(100);
}

PointPillar::~PointPillar(void)
{
  trt_.reset();

  checkCudaErrors(cudaFree(box_output));
  checkCudaErrors(cudaFree(box_num));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
}
// getPointSize
int PointPillar::getPointSize() {
  return trt_->getPointSize();
}
//doinfer函数:执行TensorRT推理,获取预测框结果
int PointPillar::doinfer(
  void*points_data,
  unsigned int* points_size,
  std::vector<Bndbox> &nms_pred,
  float nms_iou_thresh,
  int pre_nms_top_n,
  std::vector<std::string>& class_names,
  bool do_profile
)
{
#if PERFORMANCE_LOG
  float doinferTime = 0.0f;
  cudaEventRecord(start, stream_);
#endif
  void *buffers[] = {points_data, points_size, box_output, box_num};

  trt_->doinfer(buffers, do_profile);

#if PERFORMANCE_LOG
  checkCudaErrors(cudaEventRecord(stop, stream_));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&doinferTime, start, stop));
  std::cout<<"TIME: doinfer: "<< doinferTime <<" ms." <<std::endl;
#endif
  cudaDeviceSynchronize();
  int num_obj = box_num[0];
  for (int i = 0; i < num_obj; i++) {
    auto Bb = Bndbox(
      box_output[i * 9],
      box_output[i * 9 + 1],
      box_output[i * 9 + 2],
      box_output[i * 9 + 3],
      box_output[i * 9 + 4],
      box_output[i * 9 + 5],
      box_output[i * 9 + 6],
      box_output[i * 9 + 7],
      box_output[i * 9 + 8]
    );
    res.push_back(Bb);
  }
  nms_cpu(res, nms_iou_thresh, nms_pred, pre_nms_top_n);
  for(int i=0; i<nms_pred.size(); i++) {
    printf("%s, %f, %f, %f, %f, %f, %f, %f, %f\n",
      class_names[nms_pred[i].id].c_str(), nms_pred[i].x,
      nms_pred[i].y, nms_pred[i].z, nms_pred[i].l, nms_pred[i].w,
      nms_pred[i].h, nms_pred[i].rt, nms_pred[i].score);
  }
  res.clear();
return 0;
}

