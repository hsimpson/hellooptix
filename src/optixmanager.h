#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

class OptixManager {
 public:
  OptixManager();

 protected:
  void initOptix();
  void createContext();
  void createModule();

 protected:
  CUcontext      _cudaContext;
  CUstream       _stream;
  cudaDeviceProp _deviceProps;

  OptixDeviceContext _optixContext;

  OptixPipeline               _pipeline;
  OptixPipelineCompileOptions _pipelineCompileOptions = {};
  OptixPipelineLinkOptions    _pipelineLinkOptions    = {};

  OptixModule               _module;
  OptixModuleCompileOptions _moduleCompileOptions = {};
};
