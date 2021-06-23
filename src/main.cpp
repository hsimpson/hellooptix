#include "optixmanager.h"
#include <iostream>

int main() {
  try {
    OptixManager omgr;
    omgr.writeImage("./image.ppm");

  } catch (std::runtime_error& e) {
    std::cerr << "Fatal Error: " << e.what() << std::endl;
  }

  return 0;
}
