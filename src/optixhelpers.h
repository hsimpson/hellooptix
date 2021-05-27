#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <sstream>
#include <stdexcept>
#include <format>

#define CUDA_CHECK(call)                                     \
  {                                                          \
    cudaError_t rc = call;                                   \
    if (rc != cudaSuccess) {                                 \
      cudaError_t err = rc; /*cudaGetLastError();*/          \
      std::cerr << std::format("CUDA error {} ({}) ({}:{})", \
                               cudaGetErrorName(err),        \
                               cudaGetErrorString(err),      \
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
                               cudaGetErrorName(err),             \
                               cudaGetErrorString(err),           \
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
