#pragma once

#pragma once
#include <string>
#include <map>
#include <tiny_obj_loader.h>
#include "./trianglemesh.h"

class ObjLoader {
 public:
  bool load(const std::string& filename, std::shared_ptr<Scene> scene);

 private:
  int addVertex(TriangleMesh&                    triangleMesh,
                const tinyobj::attrib_t&         attrib,
                const tinyobj::index_t&          idx,
                std::map<tinyobj::index_t, int>& knownVertices);
  int loadTexture(std::shared_ptr<Scene> scene, std::map<std::string, int>& knownTextures, const std::string& filename);
};
