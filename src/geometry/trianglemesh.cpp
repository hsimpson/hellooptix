#include "trianglemesh.h"

/* the cube

     v5-----------v6
    / |          / |
   /  |         /  |
  v2----------v1   |
  |   |        |   |
  |   |        |   |
  |  v4--------|--v7
  | /          |  /
  |/           | /
  v3-----------v0

*/

Cube::Cube(glm::vec3 size, glm::vec3 center) {
  glm::vec3 halfMax = size * 0.5f + center;
  glm::vec3 halfMin = size * -0.5f + center;

  vertices.push_back({halfMax.x, halfMin.y, halfMax.z});
  vertices.push_back({halfMax.x, halfMax.y, halfMax.z});
  vertices.push_back({halfMin.x, halfMax.y, halfMax.z});
  vertices.push_back({halfMin.x, halfMin.y, halfMax.z});

  vertices.push_back({halfMin.x, halfMin.y, halfMin.z});
  vertices.push_back({halfMin.x, halfMax.y, halfMin.z});
  vertices.push_back({halfMax.x, halfMax.y, halfMin.z});
  vertices.push_back({halfMax.x, halfMin.y, halfMin.z});

  // front
  indices.push_back({0, 1, 2});
  indices.push_back({2, 3, 0});

  // left
  indices.push_back({3, 2, 5});
  indices.push_back({5, 4, 3});

  // right
  indices.push_back({7, 6, 1});
  indices.push_back({1, 0, 7});

  // top
  indices.push_back({1, 6, 5});
  indices.push_back({5, 2, 1});

  // bottom
  indices.push_back({0, 3, 4});
  indices.push_back({4, 7, 0});

  // back
  indices.push_back({4, 5, 6});
  indices.push_back({6, 7, 4});
}
