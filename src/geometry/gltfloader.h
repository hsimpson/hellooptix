#pragma once
#include <string>
#include "./trianglemesh.h"

class GltfLoader {
 public:
  bool load(const std::string& filename, Scene& triangleMeshes);
  bool loadTexture(const std::string& filename, Texture& texture);
};
