#pragma once

struct Params {
  struct {
    float4 *     accumBuffer;
    uchar4 *     colorBuffer;
    uint2        size;
    unsigned int sampleIndex;
  } frame;

  struct {
    float3 eye;
    float3 U;
    float3 V;
    float3 W;
  } camera;

  OptixTraversableHandle traversable;
};

struct RayGenData {
  float r, g, b;
};

struct TriangleMeshSBTData {
  float3  color;
  float3 *vertex;
  float3 *normal;
  float2 *texcoord;
  uint3 * index;

  bool                hasTexture;
  cudaTextureObject_t texture;
};

/*! SBT record for a raygen program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *                                     data;
};

/*! SBT record for a miss program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *                                     data;
};

/*! SBT record for a hitgroup program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  TriangleMeshSBTData                        data;
};
