#pragma once
#include <cstdint>
#include <vector>
#include <assert.h>
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

  uint32_t _width  = 0u;
  uint32_t _height = 0u;

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

  return nullptr;
}

struct CUDABuffer {
  inline CUdeviceptr d_pointer() const {
    return (CUdeviceptr)d_ptr;
  }

  //! re-size buffer to given number of bytes
  void resize(size_t size) {
    if (d_ptr) free();
    alloc(size);
  }

  //! allocate to given number of bytes
  void alloc(size_t size) {
    assert(d_ptr == nullptr);
    this->sizeInBytes = size;
    CUDA_CHECK(cudaMalloc((void**)&d_ptr, sizeInBytes));
  }

  //! free allocated memory
  void free() {
    CUDA_CHECK(cudaFree(d_ptr));
    d_ptr       = nullptr;
    sizeInBytes = 0;
  }

  template <typename T>
  void alloc_and_upload(const std::vector<T>& vt) {
    alloc(vt.size() * sizeof(T));
    upload((const T*)vt.data(), vt.size());
  }

  template <typename T>
  void upload(const T* t, size_t count) {
    assert(d_ptr != nullptr);
    assert(sizeInBytes == count * sizeof(T));
    CUDA_CHECK(cudaMemcpy(d_ptr, (void*)t,
                          count * sizeof(T), cudaMemcpyHostToDevice));
  }

  template <typename T>
  void download(T* t, size_t count) {
    assert(d_ptr != nullptr);
    assert(sizeInBytes == count * sizeof(T));
    CUDA_CHECK(cudaMemcpy((void*)t, d_ptr,
                          count * sizeof(T), cudaMemcpyDeviceToHost));
  }

  size_t sizeInBytes{0};
  void*  d_ptr{nullptr};
};