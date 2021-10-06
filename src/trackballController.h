#pragma once
#include "cameraController.h"

class TrackballController : public CameraController {
 public:
  TrackballController(std::shared_ptr<Camera> camera);
  ~TrackballController() override;

  void zoom(float delta) override;
  void resize(uint32_t width, uint32_t height) override;
  void rotate(const glm::ivec2& mousePosition) override;
  void pan(const glm::ivec2& mousePosition) override;

 private:
  const float _zoomFactor     = 1.2f;
  const float _rotationFactor = 0.001f;
  const float _panFactor      = 0.005f;
  float       _cameraEyeLookAtDistance;
  glm::vec3   _cameraUp = {0.0f, 1.0f, 0.0f};
  glm::vec3   _axis     = {0.0f, 0.0f, 0.0f};
  float       _angle    = 0.0f;
};
