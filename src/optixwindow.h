#pragma once

#include "glfwindow.h"
#include "optixmanager.h"
#include <memory>
#include "geometry/trianglemesh.h"
#include "camera.h"

class OptixWindow : public GLFWindow {
 public:
  OptixWindow(const std::string &title,
              const Scene &      scene,
              uint32_t           width,
              uint32_t           height);
  virtual ~OptixWindow();

  virtual void resize(uint32_t width, uint32_t height);
  virtual void draw();
  virtual void render();
  virtual void dolly(float offset);
  virtual void move(float offsetX, float offsetY);
  virtual void moveLookAt(float offsetX, float offsetY);
  virtual void rotate(float pitch, float yaw, float roll = 0);

 private:
  std::unique_ptr<OptixManager> _optixManager;

  GLuint _texture{0};
};