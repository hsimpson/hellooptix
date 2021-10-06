#pragma once
#include <string>
#include "./trianglemesh.h"

class GltfLoader {
 public:
  bool load(const std::string& filename, std::shared_ptr<Scene> scene);
  bool loadTexture(const std::string& filename, Texture& texture);
};
