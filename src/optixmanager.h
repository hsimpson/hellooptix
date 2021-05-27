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
  void createProgramGroup();
  void createPipeline();

 protected:
  CUcontext      _cudaContext;
  CUstream       _stream;
  cudaDeviceProp _deviceProps;

  OptixDeviceContext _optixContext;

  OptixModule               _module               = nullptr;
  OptixModuleCompileOptions _moduleCompileOptions = {};

  OptixProgramGroup _raygenProgramGroup = nullptr;
  OptixProgramGroup _missProgramGroup   = nullptr;

  OptixPipeline               _pipeline               = nullptr;
  OptixPipelineCompileOptions _pipelineCompileOptions = {};
};
