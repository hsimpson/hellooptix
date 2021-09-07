#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <sstream>
#include <stdexcept>
#include <format>
#include <iostream>
#include <glm/glm.hpp>
#include "spdlog/spdlog.h"

#define CUDA_CHECK(call)                              \
  {                                                   \
    cudaError_t rc = call;                            \
    if (rc != cudaSuccess) {                          \
      cudaError_t error = rc; /*cudaGetLastError();*/ \
      spdlog::error("CUDA error {} ({}) ({}:{})",     \
                    cudaGetErrorName(error),          \
                    cudaGetErrorString(error),        \
                    __FILE__,                         \
                    __LINE__);                        \
    }                                                 \
  }

#define CUDA_SYNC_CHECK()                              \
  {                                                    \
    cudaDeviceSynchronize();                           \
    cudaError_t error = cudaGetLastError();            \
    if (error != cudaSuccess) {                        \
      spdlog::error("CUDA sync error {} ({}) ({}:{})", \
                    cudaGetErrorName(error),           \
                    cudaGetErrorString(error),         \
                    __FILE__,                          \
                    __LINE__);                         \
    }                                                  \
  }

#define CUDA_CHECK_NOEXCEPT(call) \
  {                               \
    call;                         \
  }

#define OPTIX_CHECK(call)                                           \
  {                                                                 \
    OptixResult res = call;                                         \
    if (res != OPTIX_SUCCESS) {                                     \
      spdlog::error("Optix error ({}) failed with code {} ({}:{})", \
                    #call,                                          \
                    (int)res,                                       \
                    __FILE__,                                       \
                    __LINE__);                                      \
    }                                                               \
  }

#define PRINT(var) spdlog::info(var);

inline float3 vec3ToFloat3(const glm::vec3 &v) {
  return make_float3(v.x, v.y, v.z);
}
