#define NOMINMAX
#include "optixmanager.h"
#include "optixhelpers.h"
#include <iostream>
#include "program.h"
#include <optix_stack_size.h>
#include "optixtypes.h"
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <sstream>
#include <format>
#include <fstream>

const int32_t _WIDTH  = 512;
const int32_t _HEIGHT = 512;

OptixManager::OptixManager() {
  initOptix();
  createContext();
  createModule();
  createProgramGroup();
  createPipeline();
  createShaderBindingTable();
  launch();
}

OptixManager::~OptixManager() {
  delete _outputBuffer;

  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_shaderBindingTable.raygenRecord)));
  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_shaderBindingTable.missRecordBase)));

  OPTIX_CHECK(optixPipelineDestroy(_pipeline));
  OPTIX_CHECK(optixProgramGroupDestroy(_missProgramGroup));
  OPTIX_CHECK(optixProgramGroupDestroy(_raygenProgramGroup));

  OPTIX_CHECK(optixModuleDestroy(_module));
  OPTIX_CHECK(optixDeviceContextDestroy(_optixContext));
}

void OptixManager::initOptix() {
  std::cout << "init optix ..." << std::endl;
  // -------------------------------------------------------
  // check for available optix7 capable devices
  // -------------------------------------------------------
  CUDA_CHECK(cudaFree(0));
  int numDevices;
  CUDA_CHECK(cudaGetDeviceCount(&numDevices));
  if (numDevices == 0)
    throw std::runtime_error("no CUDA capable devices found!");
  std::cout << "found " << numDevices << " CUDA devices" << std::endl;

  // -------------------------------------------------------
  // initialize optix
  // -------------------------------------------------------
  OPTIX_CHECK(optixInit());

  std::cout << "optix successfully initialized" << std::endl;
}

static void context_log_cb(unsigned int level,
                           const char*  tag,
                           const char*  message,
                           void*) {
  fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

void OptixManager::createContext() {
  std::cout << "creating optix context ..." << std::endl;
  // for this sample, do everything on one device
  const int deviceID = 0;

  CUDA_CHECK(cudaSetDevice(deviceID));
  CUDA_CHECK(cudaStreamCreate(&_stream));

  cudaDeviceProp deviceProps;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, deviceID));
  std::cout << "running on device: " << deviceProps.name << std::endl;

  CUcontext cudaContext;
  CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
  if (cuRes != CUDA_SUCCESS)
    fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

  OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &_optixContext));
  OPTIX_CHECK(optixDeviceContextSetLogCallback(_optixContext, context_log_cb, nullptr, 4));
}

void OptixManager::createModule() {
  std::cout << "setting up module ..." << std::endl;

  OptixModuleCompileOptions moduleCompileOptions = {};

  moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  moduleCompileOptions.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  moduleCompileOptions.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  _pipelineCompileOptions                                  = {};
  _pipelineCompileOptions.usesMotionBlur                   = false;
  _pipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
  _pipelineCompileOptions.numPayloadValues                 = 2;
  _pipelineCompileOptions.numAttributeValues               = 2;
  _pipelineCompileOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
  _pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

  Program           prg("src/hellooptix.cu");
  const std::string ptxCode = prg.getPTX();

  char   log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK(optixModuleCreateFromPTX(_optixContext,
                                       &moduleCompileOptions,
                                       &_pipelineCompileOptions,
                                       ptxCode.c_str(),
                                       ptxCode.size(),
                                       log, &sizeof_log,
                                       &_module));
  if (sizeof_log > 1) PRINT(log);
}

void OptixManager::createProgramGroup() {
  std::cout << "create program group" << std::endl;

  OptixProgramGroupOptions programGroupOptions = {};  // Initialize to zeros

  OptixProgramGroupDesc raygenProgGroupDesc    = {};
  raygenProgGroupDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygenProgGroupDesc.raygen.module            = _module;
  raygenProgGroupDesc.raygen.entryFunctionName = "__raygen__draw_color";

  char   log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK(optixProgramGroupCreate(_optixContext,
                                      &raygenProgGroupDesc,
                                      1,
                                      &programGroupOptions,
                                      log,
                                      &sizeof_log,
                                      &_raygenProgramGroup));
  if (sizeof_log > 1) PRINT(log);

  // Leave miss group's module and entryfunc name null
  OptixProgramGroupDesc missProgGroupDesc = {};
  missProgGroupDesc.kind                  = OPTIX_PROGRAM_GROUP_KIND_MISS;
  OPTIX_CHECK(optixProgramGroupCreate(_optixContext,
                                      &missProgGroupDesc,
                                      1,
                                      &programGroupOptions,
                                      log,
                                      &sizeof_log,
                                      &_missProgramGroup));
  if (sizeof_log > 1) PRINT(log);
}

void OptixManager::createPipeline() {
  std::cout << "create pipeline" << std::endl;

  const uint32_t    maxTraceDepth   = 0;
  OptixProgramGroup programGroups[] = {_raygenProgramGroup};

  OptixPipelineLinkOptions pipelineLinkOptions = {};
  pipelineLinkOptions.maxTraceDepth            = maxTraceDepth;
  pipelineLinkOptions.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

  char   log[2048];
  size_t sizeof_log = sizeof(log);

  OPTIX_CHECK(optixPipelineCreate(_optixContext,
                                  &_pipelineCompileOptions,
                                  &pipelineLinkOptions,
                                  programGroups,
                                  sizeof(programGroups) / sizeof(programGroups[0]),
                                  log,
                                  &sizeof_log,
                                  &_pipeline));
  //if (sizeof_log > 1) PRINT(log);

  OptixStackSizes stackSizes = {};
  for (auto& programGroup : programGroups) {
    OPTIX_CHECK(optixUtilAccumulateStackSizes(programGroup, &stackSizes));
  }

  uint32_t directCallableStackSizeFromTraversal;
  uint32_t directCallableStackSizeFromState;
  uint32_t continuationStackSize;
  OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes,
                                         maxTraceDepth,
                                         0,  // maxCCDepth
                                         0,  // maxDCDEpth
                                         &directCallableStackSizeFromTraversal,
                                         &directCallableStackSizeFromState,
                                         &continuationStackSize));

  OPTIX_CHECK(optixPipelineSetStackSize(_pipeline,
                                        directCallableStackSizeFromTraversal,
                                        directCallableStackSizeFromState,
                                        continuationStackSize,
                                        2  // maxTraversableDepth
                                        ));
}

void OptixManager::createShaderBindingTable() {
  CUdeviceptr  raygenRecord;
  const size_t raygenRecordSize = sizeof(RayGenSbtRecord);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygenRecord), raygenRecordSize));
  RayGenSbtRecord raygenShaderBindingTableRecord;
  OPTIX_CHECK(optixSbtRecordPackHeader(_raygenProgramGroup, &raygenShaderBindingTableRecord));
  raygenShaderBindingTableRecord.data = {0.462f, 0.725f, 0.f};
  CUDA_CHECK(cudaMemcpy(
      reinterpret_cast<void*>(raygenRecord),
      &raygenShaderBindingTableRecord,
      raygenRecordSize,
      cudaMemcpyHostToDevice));

  CUdeviceptr  missRecord;
  const size_t missRecordSize = sizeof(MissSbtRecord);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&missRecord), missRecordSize));
  RayGenSbtRecord missShaderBindingTableRecord;
  OPTIX_CHECK(optixSbtRecordPackHeader(_missProgramGroup, &missShaderBindingTableRecord));
  CUDA_CHECK(cudaMemcpy(
      reinterpret_cast<void*>(missRecord),
      &missShaderBindingTableRecord,
      missRecordSize,
      cudaMemcpyHostToDevice));

  _shaderBindingTable.raygenRecord            = raygenRecord;
  _shaderBindingTable.missRecordBase          = missRecord;
  _shaderBindingTable.missRecordStrideInBytes = sizeof(MissSbtRecord);
  _shaderBindingTable.missRecordCount         = 1;
}

void OptixManager::launch() {
  _outputBuffer = new CUDAOutputBuffer<uchar4>(CUDAOutputBufferType::CUDA_DEVICE, _WIDTH, _HEIGHT);

  CUstream stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  Params params;
  params.image      = _outputBuffer->map();
  params.imageWidth = _WIDTH;

  CUdeviceptr dParam;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dParam), sizeof(Params)));
  CUDA_CHECK(cudaMemcpy(
      reinterpret_cast<void*>(dParam),
      &params,
      sizeof(params),
      cudaMemcpyHostToDevice));

  OPTIX_CHECK(optixLaunch(_pipeline, _stream, dParam, sizeof(Params), &_shaderBindingTable, _WIDTH, _HEIGHT, /*depth=*/1));

  CUDA_SYNC_CHECK();
  _outputBuffer->unmap();
}

void OptixManager::writeImage(const std::string& imagePath) {
  // write the image as PPM
  auto imageData = _outputBuffer->getHostPointer();

  std::ofstream ofStream(imagePath, std::ios::out);
  if (ofStream.good()) {
    // header
    ofStream << "P3" << std::endl;
    ofStream << _WIDTH << " " << _HEIGHT << std::endl;
    ofStream << "255" << std::endl;

    // pixel data
    for (int32_t i = 0; i < _HEIGHT; i++) {
      for (int32_t j = 0; j < _WIDTH; j++) {
        int32_t idx = i * _WIDTH + j;
        ofStream << std::format("{} {} {}",
                                imageData[idx].x,
                                imageData[idx].y,
                                imageData[idx].z)
                 << std::endl;
      }
    }
  }
}
