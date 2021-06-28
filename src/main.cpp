#include "optixwindow.h"
#include <iostream>

int main() {
  OptixWindow optixWindow("Hello Optix!", 1024, 1024);
  optixWindow.run();

  return 0;
}
