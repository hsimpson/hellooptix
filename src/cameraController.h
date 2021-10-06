
#pragma once

#include <memory>
#include "camera.h"

class CameraController {
 public:
  CameraController(std::shared_ptr<Camera> camera);
  virtual ~CameraController();

  virtual void zoom(float delta)                       = 0;
  virtual void resize(uint32_t width, uint32_t height) = 0;
  virtual void rotate(const glm::ivec2& mousePosition) = 0;
  virtual void pan(const glm::ivec2& mousePosition)    = 0;

  void setMousePosition(const glm::ivec2& mousePosition) {
    _mousePosition = mousePosition;
  }

 protected:
  std::shared_ptr<Camera> _camera;
  glm::ivec2              _mousePosition = glm::ivec2(0, 0);
};