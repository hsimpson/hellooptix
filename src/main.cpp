#include "optixwindow.h"
#include <iostream>
#include "geometry/trianglemesh.h"
#include "geometry/gltfloader.h"
#include "camera.h"

int main() {
  /*
  auto cube1 = Cube({2.0f, 2.0f, 2.0f}, {0.0f, 1.05f, 0.0f});
  // auto cube1  = Cube({2.0f, 2.0f, 2.0f});
  cube1.color = {0.0f, 1.0f, 0.0f};
  auto cube2  = Cube({10.0f, 0.1f, 10.0f});
  cube2.color = {1.0f, 0.0f, 0.0f};

  std::vector<TriangleMesh> meshes = {cube1, cube2};
  */

  Model model;

  GltfLoader loader;
  // loader.load("./src/assets/models/cube.gltf", model);
  // loader.load("./src/assets/models/monkey.gltf", model);
  // loader.load("./src/assets/models/torus.gltf", model);
  loader.load("./src/assets/models/Citadell.gltf", model);

  Camera camera = {
      {45.0f, 30.0f, 10.0f},
      {0.0f, 20.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      70.0f};

  OptixWindow optixWindow("Hello Optix!", model, camera, 1920, 1080);
  optixWindow.run();

  return 0;
}
