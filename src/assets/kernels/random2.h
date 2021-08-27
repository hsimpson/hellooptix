#pragma once

struct RNG {
  inline __device__ RNG() {
    //
  }

  inline __device__ void init(unsigned int seed) {
    _seed = seed;
  }

  inline __device__ unsigned int tauStep(unsigned int z, int s1, int s2, int s3, unsigned int M) {
    unsigned int b = (((z << s1) ^ z) >> s2);
    return z       = (((z & M) << s3) ^ b);
  }

  inline __device__ float operator()() {
    unsigned int z1, z2, z3, z4, r;
    z1    = tauStep(_seed, 13, 19, 12, 429496729);
    z2    = tauStep(_seed, 2, 25, 4, 4294967288);
    z3    = tauStep(_seed, 3, 11, 17, 429496280);
    z4    = (1664525 * _seed + 1013904223);
    _seed = (z1 ^ z2 ^ z3 ^ z4);
    return float(_seed) * 2.3283064365387e-10;
    return 1.0;
  }

  unsigned int _seed;
};
