#include <optix.h>
#include <cuda_runtime.h>

#include "optixtypes.h"

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__draw_color() {
  uint3       launch_index                                          = optixGetLaunchIndex();
  RayGenData* rtData                                                = (RayGenData*)optixGetSbtDataPointer();
  params.image[launch_index.y * params.imageWidth + launch_index.x] = make_uchar4(255u, 0, 0u, 255u);
}
