#include "optixwindow.h"
#include <iostream>
#include "geometry/trianglemesh.h"

int main() {
  TriangleMesh model = Cube(glm::vec3(2.0f, 3.0f, 4.0f), glm::vec3(1.0f, 1.0f, 1.0f));
  // TriangleMesh model = Cube();

  OptixWindow optixWindow("Hello Optix!", 1024, 1024);
  optixWindow.run();

  return 0;
}
