#include "optixwindow.h"
#include <iostream>
#include "geometry/trianglemesh.h"
#include "camera.h"

int main() {
  auto cube1  = Cube({2.0f, 2.0f, 2.0f}, {0.0f, 1.05f, 0.0f});
  cube1.color = {0.0f, 1.0f, 0.0f};
  auto cube2  = Cube({10.0f, 0.1f, 10.0f});
  cube2.color = {1.0f, 0.0f, 0.0f};

  std::vector<TriangleMesh> meshes = {cube1, cube2};

  Camera camera = {
      {15.0f, 5.0f, 20.0f},
      {0.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      20.0f};

  OptixWindow optixWindow("Hello Optix!", meshes, camera, 1024, 1024);
  optixWindow.run();

  return 0;
}
