#pragma once
#include "cameraController.h"

class TrackballController : public CameraController {
 public:
  TrackballController(std::shared_ptr<Camera> camera);
  ~TrackballController() override;

  void setReferenceFrame(const glm::vec3& u, const glm::vec3& v, const glm::vec3& w);
  void setGimbalLock(bool gimbalLock) { _gimbalLock = gimbalLock; }
  void reinitOrientationFromCamera();
  void zoom(float delta) override;
  void resize(uint32_t width, uint32_t height) override;
  void updateTracking(const glm::ivec2& mousePosition) override;

 private:
  void updateCamera();

  bool        _gimbalLock      = false;
  const float _zoomFactor      = 1.2f;
  const float _trackBallFactor = 0.1f;
  float       _cameraEyeLookAtDistance;

  float _latitude  = 0.0f;
  float _longitude = 0.0f;

  glm::vec3 _u = {0.0f, 0.0f, 0.0f};
  glm::vec3 _v = {0.0f, 0.0f, 0.0f};
  glm::vec3 _w = {0.0f, 0.0f, 0.0f};
};
