#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include "cudaoutputbuffer.h"
#include "optixtypes.h"
#include "camera.h"
#include "geometry/trianglemesh.h"
#include "cameraController.h"

class OptixManager {
 public:
  OptixManager(std::shared_ptr<Scene> scene,
               uint32_t               width,
               uint32_t               height);
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
  void createRayGenProgramGroup();
  void createMissProgramGroup();
  void createHitProgramGroup();
  void buildAccel();
  void createPipeline();
  void createTextures();
  void createShaderBindingTable();

  void updateCamera();

  uint32_t _width           = 0;
  uint32_t _height          = 0;
  float    _dollyZoomOffset = 1.0f;

  CUstream _stream;

  OptixDeviceContext _optixContext;

  OptixModule _module = nullptr;

  std::vector<OptixProgramGroup> _raygenProgramGroups;
  CUDABuffer                     _raygenRecordsBuffer;
  std::vector<OptixProgramGroup> _missProgramGroups;
  CUDABuffer                     _missRecordsBuffer;
  std::vector<OptixProgramGroup> _hitProgramGroups;
  CUDABuffer                     _hitRecordsBuffer;
  OptixShaderBindingTable        _shaderBindingTable = {};

  OptixPipeline               _pipeline               = nullptr;
  OptixPipelineCompileOptions _pipelineCompileOptions = {};

  CUDAOutputBuffer<uchar4>* _outputBuffer = nullptr;
  CUDAOutputBuffer<float4>* _accumBuffer  = nullptr;
  Params                    _launchParams;

  std::shared_ptr<Scene> _scene;

  // one buffer per input mesh
  std::vector<CUDABuffer> _vertexBuffer;
  std::vector<CUDABuffer> _normalBuffer;
  std::vector<CUDABuffer> _texcoordBuffer;
  std::vector<CUDABuffer> _indexBuffer;

  // buffer that keeps the (final, compacted) accel structure
  CUDABuffer _asBuffer;

  // one texture object and pixel array per used texture
  std::vector<cudaArray_t>         _textureArrays;
  std::vector<cudaTextureObject_t> _textureObjects;
};
