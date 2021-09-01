#include "./objloader.h"
#include <iostream>
#include <filesystem>
#include <set>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

// #define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace std {
inline bool operator<(const tinyobj::index_t& a,
                      const tinyobj::index_t& b) {
  if (a.vertex_index < b.vertex_index) return true;
  if (a.vertex_index > b.vertex_index) return false;

  if (a.normal_index < b.normal_index) return true;
  if (a.normal_index > b.normal_index) return false;

  if (a.texcoord_index < b.texcoord_index) return true;
  if (a.texcoord_index > b.texcoord_index) return false;

  return false;
}
}  // namespace std

int ObjLoader::addVertex(TriangleMesh&                    triangleMesh,
                         const tinyobj::attrib_t&         attrib,
                         const tinyobj::index_t&          idx,
                         std::map<tinyobj::index_t, int>& knownVertices) {
  if (knownVertices.find(idx) != knownVertices.end())
    return knownVertices[idx];

  const glm::vec3* vertex_array   = (const glm::vec3*)attrib.vertices.data();
  const glm::vec3* normal_array   = (const glm::vec3*)attrib.normals.data();
  const glm::vec2* texcoord_array = (const glm::vec2*)attrib.texcoords.data();

  int newID          = (int)triangleMesh.vertices.size();
  knownVertices[idx] = newID;

  triangleMesh.vertices.push_back(vertex_array[idx.vertex_index]);
  if (idx.normal_index >= 0) {
    while (triangleMesh.normals.size() < triangleMesh.vertices.size())
      triangleMesh.normals.push_back(normal_array[idx.normal_index]);
  }
  if (idx.texcoord_index >= 0) {
    while (triangleMesh.texcoords.size() < triangleMesh.vertices.size())
      triangleMesh.texcoords.push_back(texcoord_array[idx.texcoord_index]);
  }

  // just for sanity's sake:
  if (triangleMesh.texcoords.size() > 0)
    triangleMesh.texcoords.resize(triangleMesh.vertices.size());
  // just for sanity's sake:
  if (triangleMesh.normals.size() > 0)
    triangleMesh.normals.resize(triangleMesh.vertices.size());

  return newID;
}

int ObjLoader::loadTexture(Scene& scene, std::map<std::string, int>& knownTextures, const std::string& filename) {
  if (knownTextures.find(filename) != knownTextures.end())
    return knownTextures[filename];

  int resolutionX = 0;
  int resolutionY = 0;
  int components  = 0;
  stbi_set_flip_vertically_on_load(true);  // flip the image vertically
  unsigned char* imageData = stbi_load(filename.c_str(), &resolutionX, &resolutionY, &components, STBI_rgb_alpha);

  // set components fixed to 4 because we want to use RGBA
  components = 4;

  int textureID = -1;
  if (imageData) {
    textureID = scene.textures.size();
    Texture texture;
    texture.resolution      = glm::uvec2(resolutionX, resolutionY);
    const unsigned int size = resolutionX * resolutionY * components;

    texture.pixelData.resize(size);
    std::copy(imageData, imageData + resolutionX * resolutionY * components, texture.pixelData.begin());

    free(imageData);

    scene.textures.push_back(texture);
  }
  knownTextures[filename] = textureID;
  return textureID;
}

bool ObjLoader::load(const std::string& filename, Scene& scene) {
  // get parent directory of obj file
  std::filesystem::path path(filename);
  auto                  basePath = path.parent_path();

  tinyobj::ObjReaderConfig reader_config;
  tinyobj::ObjReader       reader;
  reader_config.mtl_search_path = basePath.string();  // Path to material files

  if (!reader.ParseFromFile(filename, reader_config)) {
    if (!reader.Error().empty()) {
      std::cerr << "TinyObjReader: " << reader.Error();
    }
    return false;
  }

  if (!reader.Warning().empty()) {
    std::cout << "TinyObjReader: " << reader.Warning();
  }

  if (!reader.Error().empty()) {
    std::cerr << "TinyObjReader: " << reader.Error();
  }

  auto& attrib    = reader.GetAttrib();
  auto& shapes    = reader.GetShapes();
  auto& materials = reader.GetMaterials();

  std::map<std::string, int> knownTextures;

  for (const auto& shape : shapes) {
    std::set<int> materialIDs;
    for (auto faceMatID : shape.mesh.material_ids)
      materialIDs.insert(faceMatID);

    std::map<tinyobj::index_t, int> knownVertices;

    for (int materialID : materialIDs) {
      TriangleMesh triangleMesh;

      for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
        if (shape.mesh.material_ids[faceID] != materialID) continue;
        tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
        tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
        tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

        glm::uvec3 idx = {
            addVertex(triangleMesh, attrib, idx0, knownVertices),
            addVertex(triangleMesh, attrib, idx1, knownVertices),
            addVertex(triangleMesh, attrib, idx2, knownVertices)};

        triangleMesh.indices.push_back(idx);
        const auto diffuseColor = materials[materialID].diffuse;
        triangleMesh.color      = {diffuseColor[0], diffuseColor[1], diffuseColor[2]};

        if (materials[materialID].diffuse_texname.length() > 0) {
          const auto& textureFilename = basePath / materials[materialID].diffuse_texname;
          triangleMesh.textureID      = loadTexture(scene, knownTextures, textureFilename.string());
        }
      }

      triangleMesh.boundingBox.addPoints(triangleMesh.vertices);
      scene.meshes.push_back(triangleMesh);
      scene.boundingBox.addBox(triangleMesh.boundingBox);
    }
  }

  return true;
}
