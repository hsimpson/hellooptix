#pragma once
#include <cstdint>
#include <vector>
#include "optixhelpers.h"

enum class CUDAOutputBufferType {
  CUDA_DEVICE = 0,  // not preferred, typically slower than ZERO_COPY
                    // GL_INTEROP  = 1, // single device only, preferred for single device
                    // ZERO_COPY   = 2, // general case, preferred for multi-gpu if not fully nvlink connected
                    // CUDA_P2P    = 3  // fully connected only, preferred for fully nvlink connected
};

template <typename PIXEL_FORMAT>
class CUDAOutputBuffer {
 public:
  CUDAOutputBuffer(CUDAOutputBufferType type, int32_t width, int32_t height);
  ~CUDAOutputBuffer();

  void          resize(int32_t width, int32_t height);
  PIXEL_FORMAT* map();
  void          unmap();
  PIXEL_FORMAT* getHostPointer();

 private:
  void ensureMinimumSize(int& width, int& height);
  void makeCurrent() {
    CUDA_CHECK(cudaSetDevice(_deviceIdx));
  }

  CUDAOutputBufferType _type;

  int32_t _width  = 0u;
  int32_t _height = 0u;

  PIXEL_FORMAT*             _devicePixels = nullptr;
  std::vector<PIXEL_FORMAT> _hostPixels;

  int32_t  _deviceIdx = 0;
  CUstream _stream    = 0u;
};

template <typename PIXEL_FORMAT>
CUDAOutputBuffer<PIXEL_FORMAT>::CUDAOutputBuffer(CUDAOutputBufferType type, int32_t width, int32_t height)
    : _type(type) {
  // ensureMinimumSize(width, height);

  resize(width, height);
}

template <typename PIXEL_FORMAT>
CUDAOutputBuffer<PIXEL_FORMAT>::~CUDAOutputBuffer() {
  makeCurrent();
  if (_type == CUDAOutputBufferType::CUDA_DEVICE) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_devicePixels)));
  }
}

template <typename PIXEL_FORMAT>
void CUDAOutputBuffer<PIXEL_FORMAT>::ensureMinimumSize(int& width, int& height) {
  if (width <= 0) {
    width = 1;
  }

  if (height <= 0) {
    height = 1;
  }
}

template <typename PIXEL_FORMAT>
PIXEL_FORMAT* CUDAOutputBuffer<PIXEL_FORMAT>::map() {
  if (_type == CUDAOutputBufferType::CUDA_DEVICE) {
    // nothing needed
  }
  return _devicePixels;
}

template <typename PIXEL_FORMAT>
void CUDAOutputBuffer<PIXEL_FORMAT>::unmap() {
  makeCurrent();

  if (_type == CUDAOutputBufferType::CUDA_DEVICE) {
    CUDA_CHECK(cudaStreamSynchronize(_stream));
  }
}

template <typename PIXEL_FORMAT>
void CUDAOutputBuffer<PIXEL_FORMAT>::resize(int32_t width, int32_t height) {
  ensureMinimumSize(width, height);

  if (_width == width && _height == height) {
    return;
  }

  _width  = width;
  _height = height;

  makeCurrent();

  if (_type == CUDAOutputBufferType::CUDA_DEVICE) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_devicePixels)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_devicePixels),
                          _width * _height * sizeof(PIXEL_FORMAT)));
  }

  if (!_hostPixels.empty()) {
    _hostPixels.resize(_width * _height);
  }
}

template <typename PIXEL_FORMAT>
PIXEL_FORMAT* CUDAOutputBuffer<PIXEL_FORMAT>::getHostPointer() {
  if (_type == CUDAOutputBufferType::CUDA_DEVICE) {
    _hostPixels.resize(_width * _height);
    makeCurrent();
    CUDA_CHECK(cudaMemcpy(
        static_cast<void*>(_hostPixels.data()),
        map(),
        _width * _height * sizeof(PIXEL_FORMAT),
        cudaMemcpyDeviceToHost));
    unmap();
    return _hostPixels.data();
  }
}
