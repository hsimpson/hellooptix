#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <memory>
#include "../camera.h"

struct Texture {
  glm::uvec2                 resolution;
  std::vector<unsigned char> pixelData;
};

struct TriangleMesh {
  std::vector<glm::vec3>  vertices;
  std::vector<glm::vec3>  normals;
  std::vector<glm::vec2>  texcoords;
  std::vector<glm::uvec3> indices;
  glm::vec3               color;

  int textureID{-1};
};

struct Scene {
  std::vector<TriangleMesh> meshes;
  std::vector<Texture>      textures;
  std::shared_ptr<Camera>   camera;
};

class Cube : public TriangleMesh {
 public:
  Cube(glm::vec3 size = {1.0f, 1.0f, 1.0f}, glm::vec3 center = {0.0f, 0.0f, 0.0f});
};
