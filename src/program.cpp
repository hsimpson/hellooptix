#include <iostream>
#include <fstream>
#include <vector>
#include <nvrtc.h>
#include "program.h"
#include "nvrtchelpers.h"
#include <config.h>
#include "spdlog/spdlog.h"

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
      "-D__x86_64",        \
      "--device-debug"

Program::Program(std::string filename)
    : _filename(filename) {
}

const std::string Program::getPTX() {
  std::ifstream inputFile(_filename.c_str(), std::ios::binary);
  if (!inputFile.good()) {
    spdlog::error("Failed to load file {}", _filename);
    return "";
  }
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
  const char*              abs_dirs[] = {NVRTC_ABSOLUTE_INCLUDE_DIRS};
  const char*              rel_dirs[] = {NVRTC_RELATIVE_INCLUDE_DIRS};

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

      spdlog::error("NVRTC compilation failed: {}", compileLog);
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
