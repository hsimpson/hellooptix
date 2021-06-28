#pragma once

struct Params {
  uchar4*      image;
  unsigned int imageWidth;
  unsigned int imageHeight;
};

struct RayGenData {
  float r, g, b;
};

template <typename T>
struct SbtRecord {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

using RayGenSbtRecord = SbtRecord<RayGenData>;
using MissSbtRecord   = SbtRecord<int>;
