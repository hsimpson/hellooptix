#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include "cudaoutputbuffer.h"

class OptixManager {
 public:
  OptixManager();
  ~OptixManager();

  void writeImage(const std::string& imagePath);

 protected:
  void initOptix();
  void createContext();
  void createModule();
  void createProgramGroup();
  void createPipeline();
  void createShaderBindingTable();
  void launch();

 protected:
  CUstream _stream;

  OptixDeviceContext _optixContext;

  OptixModule _module = nullptr;

  OptixProgramGroup _raygenProgramGroup = nullptr;
  OptixProgramGroup _missProgramGroup   = nullptr;

  OptixPipeline               _pipeline               = nullptr;
  OptixPipelineCompileOptions _pipelineCompileOptions = {};

  OptixShaderBindingTable _shaderBindingTable = {};

  CUDAOutputBuffer<uchar4>* _outputBuffer = nullptr;
};
