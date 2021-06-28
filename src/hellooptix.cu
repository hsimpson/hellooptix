#include <optix.h>
#include <cuda_runtime.h>

#include "optixtypes.h"

extern "C" {
__constant__ Params params;
}

__device__ float vdot(float2 v1, float2 v2) {
  return (v1.x * v2.x) + (v1.y * v2.y);
}

__device__ float fract(float f) {
  return f - floor(f);
}

__device__ float rand(float2 co) {
  return fract(sinf(vdot(co, make_float2(12.9898f, 78.233f))) * 43758.5453f);
}

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
