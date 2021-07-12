#pragma once

#include <vector>
#include <glm/glm.hpp>

class TriangleMesh {
 public:
  std::vector<glm::vec3>  vertices;
  std::vector<glm::uvec3> indices;
  glm::vec3               color;
};

class Cube : public TriangleMesh {
 public:
  Cube(glm::vec3 size = {1.0f, 1.0f, 1.0f}, glm::vec3 center = {0.0f, 0.0f, 0.0f});
};
