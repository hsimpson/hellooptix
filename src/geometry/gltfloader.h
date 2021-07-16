#pragma once
#include <string>
#include "./trianglemesh.h"

class GltfLoader {
 public:
  GltfLoader();
  bool load(const std::string& filename, Model& triangleMeshes);
};