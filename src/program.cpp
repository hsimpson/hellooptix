#include <iostream>
#include <fstream>
#include <vector>
#include <nvrtc.h>
#include "program.h"
#include "nvrtchelpers.h"

// TODO: make this dynamic (environement variable(s))
#define SAMPLES_ABSOLUTE_INCLUDE_DIRS                          \
  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0/include", \
      "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include",

// NVRTC compiler options
#define CUDA_NVRTC_OPTIONS \
  "-std=c++11",            \
      "-arch",             \
      "compute_60",        \
      "-use_fast_math",    \
      "-lineinfo",         \
      "-default-device",   \
      "-rdc",              \
      "true",              \
      "-D__x86_64",

Program::Program(std::string filename)
    : _filename(filename) {
}

const std::string Program::getPTX() {
  std::ifstream inputFile(_filename.c_str(), std::ios::binary);
  if (inputFile.good()) {
    std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(inputFile), {});
    std::string                cudaCode;
    cudaCode.assign(buffer.begin(), buffer.end());

    // Create program
    nvrtcProgram prog = 0;
    NVRTC_CHECK_ERROR(nvrtcCreateProgram(&prog, cudaCode.c_str(), _filename.c_str(), 0, NULL, NULL));

    // NVRTC options
    std::vector<const char*> options;

    // Collect include dirs
    std::vector<std::string> include_dirs;
    const char*              abs_dirs[] = {SAMPLES_ABSOLUTE_INCLUDE_DIRS};
    //const char*              rel_dirs[] = {SAMPLES_RELATIVE_INCLUDE_DIRS};

    for (const char* dir : abs_dirs) {
      include_dirs.push_back(std::string("-I") + dir);
    }
    /*
    for (const char* dir : rel_dirs) {
      include_dirs.push_back("-I" + base_dir + '/' + dir);
    }
    */
    for (const std::string& dir : include_dirs) {
      options.push_back(dir.c_str());
    }

    // Collect NVRTC options
    const char* compiler_options[] = {CUDA_NVRTC_OPTIONS};
    std::copy(std::begin(compiler_options), std::end(compiler_options), std::back_inserter(options));

    // JIT compile CU to PTX
    const nvrtcResult compileRes = nvrtcCompileProgram(prog, (int)options.size(), options.data());

    if (compileRes != NVRTC_SUCCESS) {
      size_t logSize = 0;
      NVRTC_CHECK_ERROR(nvrtcGetProgramLogSize(prog, &logSize));

      std::string compileLog;
      compileLog.resize(logSize);

      if (logSize > 1) {
        NVRTC_CHECK_ERROR(nvrtcGetProgramLog(prog, &compileLog[0]));

        std::cerr << "NVRTC compile failed:" << std::endl
                  << compileLog;
      }
      return "";
    }

    // Retrieve PTX code
    size_t      ptx_size = 0;
    std::string ptx;
    NVRTC_CHECK_ERROR(nvrtcGetPTXSize(prog, &ptx_size));
    ptx.resize(ptx_size);
    NVRTC_CHECK_ERROR(nvrtcGetPTX(prog, &ptx[0]));

    // Cleanup
    NVRTC_CHECK_ERROR(nvrtcDestroyProgram(&prog));

    return ptx;
  }

  return "";
}
