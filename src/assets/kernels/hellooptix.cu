#include <optix.h>
#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include "optixtypes.h"

extern "C" __constant__ Params optixLaunchParams;

enum {
  SURFACE_RAY_TYPE = 0,
  RAY_TYPE_COUNT   = 1
};

static __forceinline__ __device__ void setPayload(float3 p) {
  optixSetPayload_0(float_as_int(p.x));
  optixSetPayload_1(float_as_int(p.y));
  optixSetPayload_2(float_as_int(p.z));
}

extern "C" __global__ void __closesthit__radiance() {
  const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData *)optixGetSbtDataPointer();

  // get basic hit information
  const int   primID = optixGetPrimitiveIndex();
  const uint3 index  = sbtData.index[primID];
  const float u      = optixGetTriangleBarycentrics().x;
  const float v      = optixGetTriangleBarycentrics().y;

  // compute normal
  float3 N;
  if (sbtData.normal) {
    N = (1.0f - u - v) * sbtData.normal[index.x] + u * sbtData.normal[index.y] + v * sbtData.normal[index.z];
  } else {
    const float3 &A = sbtData.vertex[index.x];
    const float3 &B = sbtData.vertex[index.y];
    const float3 &C = sbtData.vertex[index.z];
    N               = cross(B - A, C - A);
  }

  N = normalize(N);

  const float3 rayDir = optixGetWorldRayDirection();
  const float  cosDN  = 0.2f + 0.8f * fabsf(dot(rayDir, N));

  float3 diffuseColor = sbtData.color;

  if (sbtData.hasTexture && sbtData.texcoord) {
    const float2 tc = (1.f - u - v) * sbtData.texcoord[index.x] + u * sbtData.texcoord[index.y] + v * sbtData.texcoord[index.z];

    float4 fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
    // diffuseColor *= fromTexture;
    diffuseColor = make_float3(fromTexture.x, fromTexture.y, fromTexture.z);
  }

  const float3 color = cosDN * diffuseColor;
  setPayload(color);
}

extern "C" __global__ void __anyhit__radiance() {
  // do nothing for the moment
}

extern "C" __global__ void __miss__radiance() {
  // set a constant background color
  // const float3 bgColor = make_float3(1.0f, 1.0f, 1.0f);  // white
  // const float3 bgColor = make_float3(1.0f, 0.0f, 0.0f);  // red
  const float3 bgColor = make_float3(0.7f, 0.8f, 1.0f);
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
