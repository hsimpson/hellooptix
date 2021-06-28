#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include "cudaoutputbuffer.h"
#include "optixtypes.h"

class OptixManager {
 public:
  OptixManager(uint32_t width, uint32_t height);
  ~OptixManager();

  void                      launch();
  void                      writeImage(const std::string& imagePath);
  void                      resize(uint32_t width, uint32_t height);
  CUDAOutputBuffer<uchar4>* getOutputBuffer() {
    return _outputBuffer;
  }

 private:
  void initOptix();
  void createContext();
  void createModule();
  void createProgramGroup();
  void createPipeline();
  void createShaderBindingTable();

  uint32_t _width  = 0;
  uint32_t _height = 0;

  CUstream _stream;

  OptixDeviceContext _optixContext;

  OptixModule _module = nullptr;

  OptixProgramGroup _raygenProgramGroup = nullptr;
  OptixProgramGroup _missProgramGroup   = nullptr;

  OptixPipeline               _pipeline               = nullptr;
  OptixPipelineCompileOptions _pipelineCompileOptions = {};

  OptixShaderBindingTable _shaderBindingTable = {};

  CUDAOutputBuffer<uchar4>* _outputBuffer = nullptr;
  Params                    _launchParams;
};
