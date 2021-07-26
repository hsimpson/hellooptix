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

  Scene scene;

  GltfLoader loader;
  // loader.load("./src/assets/models/cube.gltf", scene);
  // loader.load("./src/assets/models/monkey.gltf", scene);
  // loader.load("./src/assets/models/torus.gltf", scene);
  loader.load("./src/assets/models/scene.gltf", scene);

  if (!scene.camera) {
    scene.camera = std::make_shared<Camera>(Camera(
        {2.0f, 4.0f, 4.0f},
        {0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        glm::radians(30.0f)

            ));
  }

  OptixWindow optixWindow("Hello Optix!", scene, 1920, 1080);
  optixWindow.run();

  return 0;
}
