#include "optixwindow.h"
#include <iostream>
#include "geometry/trianglemesh.h"
#include "geometry/gltfloader.h"
#include "geometry/objloader.h"
#include "camera.h"
#include "trackballController.h"
#include "spdlog/spdlog.h"
#include "spdlog/stopwatch.h"

int main() {
  /*
  auto cube1 = Cube({2.0f, 2.0f, 2.0f}, {0.0f, 1.05f, 0.0f});
  // auto cube1  = Cube({2.0f, 2.0f, 2.0f});
  cube1.color = {0.0f, 1.0f, 0.0f};
  auto cube2  = Cube({10.0f, 0.1f, 10.0f});
  cube2.color = {1.0f, 0.0f, 0.0f};

  std::vector<TriangleMesh> meshes = {cube1, cube2};
  */

  spdlog::set_level(spdlog::level::debug);  // Set global log level to debug

  spdlog::stopwatch       sw;
  std::shared_ptr<Scene>  scene  = std::make_shared<Scene>();
  std::shared_ptr<Camera> camera = std::make_shared<Camera>();
  scene->camera                  = camera;

  // default camera parameters
  constexpr uint32_t width  = 1280;
  constexpr uint32_t height = 720;
  camera->setSize({width, height});
  camera->setEye({0.0f, 0.0f, 5.0f});
  camera->setLookAt({0.0f, 0.0f, 0.0f});
  camera->setUp({0.0f, 1.0f, 0.0f});

  GltfLoader gltfLoader;
  ObjLoader  objLoader;

  // gltfLoader.load("./src/assets/models/cube.gltf", scene);
  // gltfLoader.load("./src/assets/models/monkey.gltf", scene);
  // gltfLoader.load("./src/assets/models/torus.gltf", scene);
  gltfLoader.load("./src/assets/models/scene.gltf", scene);
  // gltfLoader.load("./src/assets/models/multi-texture/multi_texture.gltf", scene);

  // restricted models
  // gltfLoader.load("./src/assets/models/restricted/Citadell/Citadell.gltf", scene);
  // gltfLoader.load("./src/assets/models/restricted/sponza-gltf/sponza.gltf", scene);
  // objLoader.load("./src/assets/models/restricted/sponza/sponza.obj", scene);
  // objLoader.load("./src/assets/models/restricted/CornellBox/CornellBox-Original.obj", scene);
  // objLoader.load("./src/assets/models/multi-texture/multi_texture.obj", scene);

  spdlog::debug("Duration scene loading: {:.3} s", sw);

  std::shared_ptr<TrackballController> trackballController =
      std::make_shared<TrackballController>(camera, scene->boundingBox.radius());
  trackballController->setReferenceFrame(
      {1.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 1.0f},
      {0.0f, 1.0f, 0.0f});
  trackballController->setGimbalLock(true);
  OptixWindow optixWindow("Hello Optix!", scene, trackballController, width, height);
  optixWindow.run();

  return 0;
}
