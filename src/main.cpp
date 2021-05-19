#include "optixmanager.h"
#include <iostream>

int main() {
  try {
    OptixManager omgr;

  } catch (std::runtime_error& e) {
    std::cerr << "Fatal Error: " << e.what() << std::endl;
  }

  return 0;
}
