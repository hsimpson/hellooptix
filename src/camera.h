#pragma once

#include <glm/glm.hpp>

struct Camera {
  glm::vec3 from;    // camera position
  glm::vec3 lookAt;  // camera look at
  glm::vec3 up;      // camera up vector
  float     fovY;    // vertical field of view in degrees
};
