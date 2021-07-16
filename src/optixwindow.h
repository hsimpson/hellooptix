#pragma once

#include "glfwindow.h"
#include "optixmanager.h"
#include <memory>
#include "geometry/trianglemesh.h"
#include "camera.h"

class OptixWindow : public GLFWindow {
 public:
  OptixWindow(const std::string &title,
              const Model &      model,
              const Camera &     camera,
              uint32_t           width,
              uint32_t           height);
  virtual ~OptixWindow();

  virtual void resize(uint32_t width, uint32_t height);
  virtual void draw();
  virtual void render();
  virtual void zoom(float offset);
  virtual void move(float offsetX, float offsetY);
  virtual void moveLookAt(float offsetX, float offsetY);

 private:
  std::unique_ptr<OptixManager> _optixManager;

  GLuint _texture{0};
};