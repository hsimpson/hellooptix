
#pragma once

#include <memory>
#include "camera.h"

class CameraController {
 public:
  CameraController(std::shared_ptr<Camera> camera);
  virtual ~CameraController();

  enum ViewMode {
    EyeFixed,
    LookAtFixed
  };

  virtual void zoom(float delta)                               = 0;
  virtual void resize(uint32_t width, uint32_t height)         = 0;
  virtual void updateTracking(const glm::ivec2& mousePosition) = 0;

  void startTracking(const glm::ivec2& mousePosition) {
    _mousePosition   = mousePosition;
    _performTracking = true;
  }

  void setViewMode(ViewMode viewMode) { _viewMode = viewMode; }

 protected:
  std::shared_ptr<Camera> _camera;
  glm::ivec2              _mousePosition   = glm::ivec2(0, 0);
  bool                    _performTracking = false;
  ViewMode                _viewMode        = LookAtFixed;
};