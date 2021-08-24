#include "boundingbox.h"

void BoundingBox::addPoint(const glm::vec3 &point) {
  if (point.x < min.x) min.x = point.x;
  if (point.y < min.y) min.y = point.y;
  if (point.z < min.z) min.z = point.z;

  if (point.x > max.x) max.x = point.x;
  if (point.y > max.y) max.y = point.y;
  if (point.z > max.z) max.z = point.z;
}

void BoundingBox::addPoints(const std::vector<glm::vec3> &points) {
  for (const auto &point : points) {
    addPoint(point);
  }
}

void BoundingBox::addBox(const BoundingBox &box) {
  if (box.min.x < min.x) min.x = box.min.x;
  if (box.min.y < min.y) min.y = box.min.y;
  if (box.min.z < min.z) min.z = box.min.z;

  if (box.max.x > max.x) max.x = box.max.x;
  if (box.max.y > max.y) max.y = box.max.y;
  if (box.max.z > max.z) max.z = box.max.z;
}

const glm::vec3 &BoundingBox::size() const {
  return max - min;
}

const float BoundingBox::radius() const {
  return glm::length(size()) / 2.0f;
}
