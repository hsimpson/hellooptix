#include "optixmanager.h"
#include "optixhelpers.h"
#include <iostream>
#include "program.h"

OptixManager::OptixManager() {
  initOptix();
  createContext();
  createModule();
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
                           const char * tag,
                           const char * message,
                           void *) {
  fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

void OptixManager::createContext() {
  std::cout << "creating optix context ..." << std::endl;
  // for this sample, do everything on one device
  const int deviceID = 0;

  CUDA_CHECK(cudaSetDevice(deviceID));
  CUDA_CHECK(cudaStreamCreate(&_stream));

  CUDA_CHECK(cudaGetDeviceProperties(&_deviceProps, deviceID));
  std::cout << "running on device: " << _deviceProps.name << std::endl;

  CUresult cuRes = cuCtxGetCurrent(&_cudaContext);
  if (cuRes != CUDA_SUCCESS)
    fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

  OPTIX_CHECK(optixDeviceContextCreate(_cudaContext, 0, &_optixContext));
  OPTIX_CHECK(optixDeviceContextSetLogCallback(_optixContext, context_log_cb, nullptr, 4));
}

void OptixManager::createModule() {
  std::cout << "setting up module ..." << std::endl;

  _moduleCompileOptions.maxRegisterCount = 50;
  _moduleCompileOptions.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  _moduleCompileOptions.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  _pipelineCompileOptions                                  = {};
  _pipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  _pipelineCompileOptions.usesMotionBlur                   = false;
  _pipelineCompileOptions.numPayloadValues                 = 2;
  _pipelineCompileOptions.numAttributeValues               = 2;
  _pipelineCompileOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
  _pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

  _pipelineLinkOptions.maxTraceDepth = 2;

  Program           prg("src/hellooptix.cu");
  const std::string ptxCode = prg.getPTX();

  char   log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK(optixModuleCreateFromPTX(_optixContext,
                                       &_moduleCompileOptions,
                                       &_pipelineCompileOptions,
                                       ptxCode.c_str(),
                                       ptxCode.size(),
                                       log, &sizeof_log,
                                       &_module));
  if (sizeof_log > 1) PRINT(log);
}
