#include <optix.h>
#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include "optixtypes.h"
#include "random.h"
// #include "random2.h"
#include "helpers.h"

extern "C" __constant__ Params optixLaunchParams;

const float EPSILON = 0.1e-4f;

enum RayType {
  SURFACE_RAY_TYPE = 0,
  RAY_TYPE_COUNT   = 1
};

struct RadiancePRD {
  float3 emitted;
  float3 radiance;
  float3 attenuation;
  float3 origin;
  float3 direction;
  // unsigned int seed;
  int countEmitted;
  int done;
  RNG rng;

  // int pad;  // padding
};

struct Onb {
  __forceinline__ __device__ Onb(const float3& normal) {
    m_normal = normal;

    if (fabs(m_normal.x) > fabs(m_normal.z)) {
      m_binormal.x = -m_normal.y;
      m_binormal.y = m_normal.x;
      m_binormal.z = 0;
    } else {
      m_binormal.x = 0;
      m_binormal.y = -m_normal.z;
      m_binormal.z = m_normal.y;
    }

    m_binormal = normalize(m_binormal);
    m_tangent  = cross(m_binormal, m_normal);
  }

  __forceinline__ __device__ void inverse_transform(float3& p) const {
    p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
  }

  float3 m_tangent;
  float3 m_binormal;
  float3 m_normal;
};

static __forceinline__ __device__ void*
                       unpackPointer(unsigned int i0, unsigned int i1) {
  const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
  void*                    ptr  = reinterpret_cast<void*>(uptr);
  return ptr;
}

static __forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0, unsigned int& i1) {
  const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
  i0                            = uptr >> 32;
  i1                            = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ RadiancePRD* getPRD() {
  const unsigned int u0 = optixGetPayload_0();
  const unsigned int u1 = optixGetPayload_1();
  return reinterpret_cast<RadiancePRD*>(unpackPointer(u0, u1));
}

static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p) {
  // Uniformly sample disk.
  const float r   = sqrtf(u1);
  const float phi = 2.0f * M_PIf * u2;
  p.x             = r * cosf(phi);
  p.y             = r * sinf(phi);

  // Project up to hemisphere.
  p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

extern "C" __global__ void __closesthit__radiance() {
  const TriangleMeshSBTData& sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

  // get basic hit information
  const float3 rayDir = optixGetWorldRayDirection();
  const int    primID = optixGetPrimitiveIndex();
  const uint3  index  = sbtData.index[primID];
  const float  u      = optixGetTriangleBarycentrics().x;
  const float  v      = optixGetTriangleBarycentrics().y;

  // compute normal
  float3 N;
  if (sbtData.normal) {
    N = (1.0f - u - v) * sbtData.normal[index.x] + u * sbtData.normal[index.y] + v * sbtData.normal[index.z];
  } else {
    const float3& A = sbtData.vertex[index.x];
    const float3& B = sbtData.vertex[index.y];
    const float3& C = sbtData.vertex[index.z];
    N               = cross(B - A, C - A);
  }

  N              = normalize(N);
  const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;

  const float cosDN = 0.2f + 0.8f * fabsf(dot(rayDir, N));

  RadiancePRD* prd = getPRD();

  if (prd->countEmitted)
    prd->emitted = make_float3(0.0f, 0.0f, 0.0f);  // emission color (used for lights)
  else
    prd->emitted = make_float3(0.0f, 0.0f, 0.0f);

  float3 diffuseColor = sbtData.color;

  if (sbtData.hasTexture && sbtData.texcoord) {
    const float2 tc = (1.f - u - v) * sbtData.texcoord[index.x] + u * sbtData.texcoord[index.y] + v * sbtData.texcoord[index.z];

    float4 fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
    // diffuseColor *= fromTexture;
    diffuseColor = make_float3(fromTexture.x, fromTexture.y, fromTexture.z);
  }

  const float3 color = cosDN * diffuseColor;

  // unsigned int seed = prd->seed;

  {
    const float z1 = prd->rng();
    const float z2 = prd->rng();

    float3 w_in;
    cosine_sample_hemisphere(z1, z2, w_in);
    Onb onb(N);
    onb.inverse_transform(w_in);
    // prd->direction = N + randomUnitVector(seed);
    prd->direction = w_in;
    prd->origin    = P + EPSILON * N;

    prd->attenuation *= color;
    prd->countEmitted = false;
  }

  const float z1 = prd->rng();
  const float z2 = prd->rng();
  // prd->seed      = seed;

  // ToDo lights
  prd->radiance = make_float3(0.0f, 0.0f, 0.0f);
}

extern "C" __global__ void __anyhit__radiance() {
  // do nothing for the moment
}

extern "C" __global__ void __miss__radiance() {
  // set a constant background color
  // const float3 bgColor = make_float3(1.0f, 1.0f, 1.0f);  // white
  // const float3 bgColor = make_float3(1.0f, 0.0f, 0.0f);  // red
  RadiancePRD* prd     = getPRD();
  const float3 bgColor = make_float3(0.7f, 0.8f, 1.0f);
  prd->radiance        = bgColor;
  prd->done            = true;
}

extern "C" __global__ void __raygen__renderFrame() {
  const unsigned int ix = optixGetLaunchIndex().x;
  const unsigned int iy = optixGetLaunchIndex().y;

  const auto& camera = optixLaunchParams.camera;

  // normalized screen plane position, in [0,1]^2
  float2 screen = make_float2(ix + .5f, iy + .5f) / make_float2(optixLaunchParams.frame.size);

  const unsigned int sampleIndex = optixLaunchParams.frame.sampleIndex;

  const int samplesPerLaunch = 16;

  float3 result = make_float3(0.0f, 0.0f, 0.0f);

  // int i = samplesPerLaunch;
  // do {
  RadiancePRD prd;
  prd.rng.init(ix + sampleIndex * optixLaunchParams.frame.size.x,
               iy + sampleIndex * optixLaunchParams.frame.size.y);
  // prd.rng.init(ix * iy * (sampleIndex * 100000));
  const float2 subpixelJitter = make_float2(prd.rng(), prd.rng());

  const float2 d = 2.0f * make_float2(
                              (static_cast<float>(ix) + subpixelJitter.x) / static_cast<float>(optixLaunchParams.frame.size.x),
                              (static_cast<float>(iy) + subpixelJitter.y) / static_cast<float>(optixLaunchParams.frame.size.y)) -
                   1.0f;

  // generate ray direction
  float3 rayDirection = normalize(d.x * camera.horizontal + d.y * camera.vertical + camera.direction);
  float3 rayOrigin    = camera.position;

  prd.emitted      = make_float3(0.0f, 0.0f, 0.0f);
  prd.radiance     = make_float3(0.0f, 0.0f, 0.0f);
  prd.attenuation  = make_float3(1.0f, 1.0f, 1.0f);
  prd.countEmitted = true;
  prd.done         = false;
  // prd.seed         = seed;

  int depth = 0;
  for (;;) {
    unsigned int p0, p1;
    packPointer(&prd, p0, p1);
    // optix trace call
    optixTrace(
        optixLaunchParams.traversable,
        rayOrigin,
        rayDirection,
        0.0f,                      // Min intersection distance
        1e16f,                     // Max intersection distance
        0.0f,                      // ray-time -- used for motion blur
        OptixVisibilityMask(255),  // Specify always visible
        OPTIX_RAY_FLAG_NONE,       // OPTIX_RAY_FLAG_NONE,
        SURFACE_RAY_TYPE,          // SBT offset
        RAY_TYPE_COUNT,            // SBT stride
        SURFACE_RAY_TYPE,          // missSBTIndex
        p0, p1);

    result += prd.emitted;
    result += prd.radiance * prd.attenuation;
    // result = prd.attenuation;

    if (prd.done || depth >= 3) {
      break;
    }

    rayOrigin    = prd.origin;
    rayDirection = prd.direction;

    ++depth;
  }

  // } while (i--);

  const unsigned int imageIndex = iy * optixLaunchParams.frame.size.x + ix;
  // float3             accumColor = result / static_cast<float>(samplesPerLaunch);
  float3 accumColor = result;

  if (sampleIndex > 0) {
    const float  a              = 1.0f / static_cast<float>(sampleIndex + 1);
    const float3 accumColorPrev = make_float3(optixLaunchParams.frame.accumBuffer[imageIndex]);
    accumColor                  = lerp(accumColorPrev, accumColor, a);
  }
  optixLaunchParams.frame.accumBuffer[imageIndex] = make_float4(accumColor, 1.0f);

  // optixLaunchParams.frame.colorBuffer[imageIndex] = make_uchar4(
  //     accumColor.x * 255u,
  //     accumColor.y * 255u,
  //     accumColor.z * 255u,
  //     255u);

  optixLaunchParams.frame.colorBuffer[imageIndex] = make_color(accumColor);
}
