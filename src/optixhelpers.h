#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <sstream>
#include <stdexcept>
#include <format>
#include <iostream>

#define CUDA_CHECK(call)                                     \
  {                                                          \
    cudaError_t rc = call;                                   \
    if (rc != cudaSuccess) {                                 \
      cudaError_t error = rc; /*cudaGetLastError();*/        \
      std::cerr << std::format("CUDA error {} ({}) ({}:{})", \
                               cudaGetErrorName(error),      \
                               cudaGetErrorString(error),    \
                               __FILE__,                     \
                               __LINE__)                     \
                << std::endl;                                \
    }                                                        \
  }

#define CUDA_SYNC_CHECK()                                         \
  {                                                               \
    cudaDeviceSynchronize();                                      \
    cudaError_t error = cudaGetLastError();                       \
    if (error != cudaSuccess) {                                   \
      std::cerr << std::format("CUDA sync error {} ({}) ({}:{})", \
                               cudaGetErrorName(error),           \
                               cudaGetErrorString(error),         \
                               __FILE__,                          \
                               __LINE__)                          \
                << std::endl;                                     \
    }                                                             \
  }

#define CUDA_CHECK_NOEXCEPT(call) \
  {                               \
    call;                         \
  }

#define OPTIX_CHECK(call)                                                      \
  {                                                                            \
    OptixResult res = call;                                                    \
    if (res != OPTIX_SUCCESS) {                                                \
      std::cerr << std::format("Optix error ({}) failed with code {} ({}:{})", \
                               #call,                                          \
                               (int)res,                                       \
                               __FILE__,                                       \
                               __LINE__)                                       \
                << std::endl;                                                  \
    }                                                                          \
  }

#define PRINT(var) std::cout << #var << "=" << var << std::endl;
