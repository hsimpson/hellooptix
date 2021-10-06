cmake_minimum_required(VERSION 3.20)
project(hellooptix 
        VERSION 0.1.0
        LANGUAGES CXX
        # LANGUAGES CXX CUDA
)



# put this in an early stage to work
# 23 --> -std:c++latest
set(CMAKE_CXX_STANDARD 20)

# include(CTest)
# enable_testing()

add_executable(hellooptix
  ./src/main.cpp
  ./src/camera.cpp
  ./src/cameraController.cpp
  ./src/trackballController.cpp
  ./src/optixmanager.cpp
  ./src/program.cpp
  ./src/glfwindow.cpp
  ./src/optixwindow.cpp
  ./src/geometry/trianglemesh.cpp
  ./src/geometry/gltfloader.cpp
  ./src/geometry/objloader.cpp
  ./src/geometry/boundingbox.cpp
)

# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)


############
# set( CMAKE_VERBOSE_MAKEFILE on )
set(TARGET_NAME "hellooptix")


set(CMAKE_MODULE_PATH
"${CMAKE_SOURCE_DIR}/CMake"
${CMAKE_MODULE_PATH}
)

if(WIN32)
  set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0" CACHE PATH "Path to OptiX installed location.")
endif()


# if (CMAKE_GENERATOR MATCHES "Visual Studio")
  # set_property(TARGET "${TARGET_NAME}" PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
# endif()

# message("CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")

find_package(CUDAToolkit REQUIRED)
find_package(OptiX REQUIRED)
# find_package(gl3w CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)


find_path(STB_INCLUDE_DIRS "stb.h")

include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${OptiX_INCLUDE}
)

find_path(GLM_INCLUDE_DIRS glm/glm.hpp)
find_path(TINYGLTF_INCLUDE_DIRS "tiny_gltf.h")

target_include_directories("${TARGET_NAME}" PRIVATE ${STB_INCLUDE_DIRS})

#target_include_directories(main PRIVATE ${TINYGLTF_INCLUDE_DIRS})

# message("OptiX_INSTALL_DIR: ${OptiX_INSTALL_DIR}")

# generate the dynamic include file
set(NVRTC_RELATIVE_INCLUDE_DIRS "\\
\".\", ")

set(NVRTC_ABSOLUTE_INCLUDE_DIRS "\\
  \"${OptiX_INCLUDE}\", \\
  \"${OptiX_INSTALL_DIR}/SDK\", \\
  \"${CUDAToolkit_INCLUDE_DIRS}\", \\
  \"${GLM_INCLUDE_DIRS}\", \\
  \"${CMAKE_SOURCE_DIR}/src\", ")

configure_file(config.h.in config.h @ONLY)

target_link_libraries( "${TARGET_NAME}"
  CUDA::cudart_static
  CUDA::nvrtc
  glad::glad
  glfw
  glm::glm
  tinyobjloader::tinyobjloader
  spdlog::spdlog
  spdlog::spdlog_header_only
)
