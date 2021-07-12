#include <optix.h>
#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include "optixtypes.h"

extern "C" __constant__ Params optixLaunchParams;

enum {
  SURFACE_RAY_TYPE = 0,
  RAY_TYPE_COUNT   = 1
};

/*
__device__ float vdot(float2 v1, float2 v2) {
  return (v1.x * v2.x) + (v1.y * v2.y);
}

__device__ float fract(float f) {
  return f - floor(f);
}

__device__ float rand(float2 co) {
  return fract(sinf(vdot(co, make_float2(12.9898f, 78.233f))) * 43758.5453f);
}
*/

/*
extern "C" __global__ void __raygen__draw_color() {
  uint3       launch_index = optixGetLaunchIndex();
  RayGenData* rtData       = (RayGenData*)optixGetSbtDataPointer();

  // float x = (float)launch_index.x;
  // float y = (float)launch_index.y;

  // float r = rand(make_float2(x + 1, y + 1));
  // float g = rand(make_float2(x + 2, y + 2));
  // float b = rand(make_float2(x + 3, y + 3));

  float r = (float)launch_index.x / (float)params.imageWidth;
  float g = (float)launch_index.y / (float)params.imageHeight;

  params.image[launch_index.y * params.imageWidth + launch_index.x] = make_uchar4(
      r * 255u,
      g * 255u,
      0u,
      255u);
}
*/

static __forceinline__ __device__ void setPayload(float3 p) {
  optixSetPayload_0(float_as_int(p.x));
  optixSetPayload_1(float_as_int(p.y));
  optixSetPayload_2(float_as_int(p.z));
}

extern "C" __global__ void __closesthit__radiance() {
  const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData *)optixGetSbtDataPointer();

  // compute normal:
  const int     primID = optixGetPrimitiveIndex();
  const uint3   index  = sbtData.index[primID];
  const float3 &A      = sbtData.vertex[index.x];
  const float3 &B      = sbtData.vertex[index.y];
  const float3 &C      = sbtData.vertex[index.z];
  const float3  Ng     = normalize(cross(B - A, C - A));

  const float3 rayDir = optixGetWorldRayDirection();
  const float  cosDN  = 0.2f + 0.8f * fabsf(dot(rayDir, Ng));

  const float3 color = cosDN * sbtData.color;
  setPayload(color);
}

extern "C" __global__ void __anyhit__radiance() {
  // do nothing for the moment
}

extern "C" __global__ void __miss__radiance() {
  // set a constant background color
  const float3 bgColor = make_float3(1.0f, 1.0f, 1.0f);  // white
  // const float3 bgColor = make_float3(1.0f, 0.0f, 0.0f);  // red
  setPayload(bgColor);
}

extern "C" __global__ void __raygen__renderFrame() {
  const unsigned int ix = optixGetLaunchIndex().x;
  const unsigned int iy = optixGetLaunchIndex().y;

  const auto &camera = optixLaunchParams.camera;

  // normalized screen plane position, in [0,1]^2
  float2 screen = make_float2(ix + .5f, iy + .5f) / make_float2(optixLaunchParams.frame.size);

  // generate ray direction
  float3 rayDirection = normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);

  unsigned int p0, p1, p2;
  // optix trace call
  optixTrace(
      optixLaunchParams.traversable,
      camera.position,
      rayDirection,
      0.0f,                           // Min intersection distance
      1e16f,                          // Max intersection distance
      0.0f,                           // ray-time -- used for motion blur
      OptixVisibilityMask(255),       // Specify always visible
      OPTIX_RAY_FLAG_DISABLE_ANYHIT,  // OPTIX_RAY_FLAG_NONE,
      SURFACE_RAY_TYPE,               // SBT offset
      RAY_TYPE_COUNT,                 // SBT stride
      SURFACE_RAY_TYPE,               // missSBTIndex
      p0, p1, p2);

  float3 result;
  result.x = int_as_float(p0);
  result.y = int_as_float(p1);
  result.z = int_as_float(p2);

  optixLaunchParams.frame.colorbuffer[iy * optixLaunchParams.frame.size.x + ix] = make_uchar4(
      result.x * 255u,
      result.y * 255u,
      result.z * 255u,
      255u);
}
