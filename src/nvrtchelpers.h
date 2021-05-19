#pragma once

#define STRINGIFY(x) STRINGIFY2(x)
#define STRINGIFY2(x) #x
#define LINE_STR STRINGIFY(__LINE__)

#define NVRTC_CHECK_ERROR(func)                                                                                 \
  do {                                                                                                          \
    nvrtcResult code = func;                                                                                    \
    if (code != NVRTC_SUCCESS)                                                                                  \
      throw std::runtime_error("ERROR: " __FILE__ "(" LINE_STR "): " + std::string(nvrtcGetErrorString(code))); \
  } while (0)
