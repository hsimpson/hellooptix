#include "gltfloader.h"

// Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "tiny_gltf.h"
#include <iostream>
#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

GltfLoader::GltfLoader() {
}

bool GltfLoader::load(const std::string& filename, Model& model) {
  tinygltf::Model    gltfModel;
  tinygltf::TinyGLTF loader;
  bool               ret = false;
  std::string        err;
  std::string        warn;

  bool isGlb = filename.ends_with(".glb");
  if (isGlb) {
    ret = loader.LoadBinaryFromFile(&gltfModel, &err, &warn, filename);  // for binary glTF(.glb)
  } else {
    ret = loader.LoadASCIIFromFile(&gltfModel, &err, &warn, filename);  // for ASCII glTF(.gltf)
  }

  if (err.length() > 0) {
    std::cerr << "Error: " << err << std::endl;
  }

  if (warn.length() > 0) {
    std::cerr << "Warning: " << warn << std::endl;
  }

  if (!ret) {
    return false;
  }

  // create the imateTextures
  for (const auto& image : gltfModel.images) {
    Texture tex;
    tex.resolution = glm::uvec2(image.width, image.height);
    tex.pixelData.resize(image.image.size());
    std::memcpy(tex.pixelData.data(), image.image.data(), image.image.size());

    model.textures.push_back(tex);
  }

  for (const auto& node : gltfModel.nodes) {
    auto mesh = gltfModel.meshes[node.mesh];

    for (auto& primitive : mesh.primitives) {
      TriangleMesh triangleMesh;

      int32_t            numOfComponents = 0;
      std::vector<float> rawFloats;

      { /* handling POSITIONS */
        const auto& accessorPositions   = gltfModel.accessors[primitive.attributes["POSITION"]];
        const auto& bufferViewPositions = gltfModel.bufferViews[accessorPositions.bufferView];
        const auto& bufferPositions     = gltfModel.buffers[bufferViewPositions.buffer];

        numOfComponents = tinygltf::GetNumComponentsInType(accessorPositions.type);

        rawFloats.resize(accessorPositions.count * numOfComponents);

        std::memcpy(rawFloats.data(), bufferPositions.data.data() + bufferViewPositions.byteOffset, bufferViewPositions.byteLength);

        for (size_t i = 0; i < rawFloats.size();) {
          glm::vec3 v = {
              rawFloats[i++],
              rawFloats[i++],
              rawFloats[i++]

          };
          triangleMesh.vertices.push_back(v);
        }
      }

      { /* handling NORMALS */
        const auto& accessorNormals   = gltfModel.accessors[primitive.attributes["NORMAL"]];
        const auto& bufferViewNormals = gltfModel.bufferViews[accessorNormals.bufferView];
        const auto& bufferNormals     = gltfModel.buffers[bufferViewNormals.buffer];

        numOfComponents = tinygltf::GetNumComponentsInType(accessorNormals.type);
        rawFloats.resize(accessorNormals.count * numOfComponents);
        std::memcpy(rawFloats.data(), bufferNormals.data.data() + bufferViewNormals.byteOffset, bufferViewNormals.byteLength);
        for (size_t i = 0; i < rawFloats.size();) {
          glm::vec3 n = {
              rawFloats[i++],
              rawFloats[i++],
              rawFloats[i++]

          };
          triangleMesh.normals.push_back(n);
        }
      }

      { /* handling TEXCOORDS */
        const auto& accessorTexCoords   = gltfModel.accessors[primitive.attributes["TEXCOORD_0"]];
        const auto& bufferViewTexCoords = gltfModel.bufferViews[accessorTexCoords.bufferView];
        const auto& bufferTexCoords     = gltfModel.buffers[bufferViewTexCoords.buffer];
        numOfComponents                 = tinygltf::GetNumComponentsInType(accessorTexCoords.type);
        rawFloats.resize(accessorTexCoords.count * numOfComponents);
        std::memcpy(rawFloats.data(), bufferTexCoords.data.data() + bufferViewTexCoords.byteOffset, bufferViewTexCoords.byteLength);
        for (size_t i = 0; i < rawFloats.size();) {
          glm::vec2 uv = {
              rawFloats[i++],
              rawFloats[i++]

          };
          triangleMesh.texcoords.push_back(uv);
        }
      }

      { /* handling INDICES */
        if (primitive.indices != -1) {
          const auto& accessorIndices   = gltfModel.accessors[primitive.indices];
          const auto& bufferViewIndices = gltfModel.bufferViews[accessorIndices.bufferView];
          const auto& bufferIndices     = gltfModel.buffers[bufferViewIndices.buffer];

          if (accessorIndices.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
            const uint16_t* indicesRaw = reinterpret_cast<const uint16_t*>(&bufferIndices.data[bufferViewIndices.byteOffset + accessorIndices.byteOffset]);

            for (size_t i = 0; i < accessorIndices.count;) {
              glm::uvec3 tri = {
                  indicesRaw[i++],
                  indicesRaw[i++],
                  indicesRaw[i++],
              };

              triangleMesh.indices.push_back(tri);
            }
          } else if (accessorIndices.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
            const uint32_t* indicesRaw = reinterpret_cast<const uint32_t*>(&bufferIndices.data[bufferViewIndices.byteOffset + accessorIndices.byteOffset]);

            for (size_t i = 0; i < accessorIndices.count;) {
              glm::uvec3 tri = {
                  indicesRaw[i++],
                  indicesRaw[i++],
                  indicesRaw[i++],
              };

              triangleMesh.indices.push_back(tri);
            }
          }
        }
      }

      { /* handling MATERIALS */
        triangleMesh.textureID = -1;
        triangleMesh.color     = {0.0f, 0.0f, 1.0f};

        if (primitive.material != -1) {
          const auto& material = gltfModel.materials[primitive.material];

          if (material.pbrMetallicRoughness.baseColorTexture.index != -1) {
            const auto& texture = gltfModel.textures[material.pbrMetallicRoughness.baseColorTexture.index];

            // const auto& image   = gltfModel.images[texture.source];
            // const auto& sampler = gltfModel.samplers[texture.sampler];

            triangleMesh.textureID = texture.sampler;
          } else if (material.pbrMetallicRoughness.baseColorFactor.size() >= 3) {
            triangleMesh.color = {
                material.pbrMetallicRoughness.baseColorFactor[0],
                material.pbrMetallicRoughness.baseColorFactor[1],
                material.pbrMetallicRoughness.baseColorFactor[2]};
          }
        }
      }

      model.meshes.push_back(triangleMesh);
    }
  }

  return ret;
}
