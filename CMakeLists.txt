cmake_minimum_required(VERSION 3.20)
project(hellooptix 
        VERSION 0.1.0
        LANGUAGES CXX
        # LANGUAGES CXX CUDA
)



# put this in an early stage to work
set(CMAKE_CXX_STANDARD 20)

include(CTest)
enable_testing()

add_executable(hellooptix
  ./src/main.cpp
  ./src/optixmanager.cpp
  ./src/program.cpp
  ./src/glfwindow.cpp
  ./src/optixwindow.cpp
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


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

message("CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")

find_package(CUDAToolkit REQUIRED)
find_package(OptiX REQUIRED)
# find_package(gl3w CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${OptiX_INCLUDE}
)

target_link_libraries( "${TARGET_NAME}"
  CUDA::cudart_static
  CUDA::nvrtc
  glad::glad
  glfw
  glm::glm
)

