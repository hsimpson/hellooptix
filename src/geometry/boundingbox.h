#pragma once

#include <glm/glm.hpp>
#include <vector>

class BoundingBox {
 public:
  void addPoint(const glm::vec3 &point);
  void addPoints(const std::vector<glm::vec3> &points);
  void addBox(const BoundingBox &box);

  const glm::vec3 size() const;
  const float     radius() const;

  glm::vec3 min = {1e10f, 1e10f, 1e10f};
  glm::vec3 max = {-1e10f, -1e10f, -1e10f};
};
