#include "trackballController.h"
#include "spdlog/spdlog.h"

TrackballController::TrackballController(std::shared_ptr<Camera> camera, float boundingRadius) : CameraController(camera) {
  _cameraEyeLookAtDistance = glm::length(_camera->lookAt() - _camera->eye());

  // _zoomFactor = 1.0f * boundingRadius;
}

TrackballController::~TrackballController() {
}

void TrackballController::zoom(float delta) {
  float zoom = delta > 0.0f ? 1.0f / _zoomFactor : _zoomFactor;
  _cameraEyeLookAtDistance *= zoom;
  spdlog::debug("Camera eye look at distance: {}", _cameraEyeLookAtDistance);
  glm::vec3 eye    = _camera->eye();
  glm::vec3 lookAt = _camera->lookAt();
  _camera->setEye(lookAt + (eye - lookAt) * zoom);
}

void TrackballController::resize(uint32_t width, uint32_t height) {
  _camera->setSize({static_cast<float>(width),
                    static_cast<float>(height)});
}

void TrackballController::updateTracking(const glm::ivec2& mousePosition) {
  if (!_performTracking) {
    startTracking(mousePosition);
    return;
  }

  glm::vec2 mouseDelta = (mousePosition - _mousePosition);
  mouseDelta *= _trackBallFactor;
  _mousePosition = mousePosition;

  spdlog::debug("TrackballController::updateTracking: {},{}", mouseDelta.x, mouseDelta.y);

  _latitude  = glm::radians(std::min(89.0f, std::max(-89.0f, glm::degrees(_latitude) + 0.5f * mouseDelta.y)));
  _longitude = glm::radians(std::fmod(glm::degrees(_longitude) - 0.5f * mouseDelta.x, 360.0f));

  updateCamera();
  if (!_gimbalLock) {
    reinitOrientationFromCamera();
    _camera->setUp(_w);
  }
}

void TrackballController::updateCamera() {
  glm::vec3 localDir;
  localDir.x = std::cos(_latitude) * std::sin(_longitude);
  localDir.y = std::cos(_latitude) * std::cos(_longitude);
  localDir.z = std::sin(_latitude);

  glm::vec3 dirWS = _u * localDir.x + _v * localDir.y + _w * localDir.z;
  if (_viewMode == EyeFixed) {
    const glm::vec3 eye = _camera->eye();
    _camera->setLookAt(eye - dirWS * _cameraEyeLookAtDistance);
  } else {
    const glm::vec3 lookAt = _camera->lookAt();
    _camera->setEye(lookAt + dirWS * _cameraEyeLookAtDistance);
  }
}

void TrackballController::setReferenceFrame(const glm::vec3& u, const glm::vec3& v, const glm::vec3& w) {
  _u = u;
  _v = v;
  _w = w;

  glm::vec3 dirWS = -glm::normalize(_camera->lookAt() - _camera->eye());
  glm::vec3 dirLocal;
  dirLocal.x = glm::dot(dirWS, _u);
  dirLocal.y = glm::dot(dirWS, _v);
  dirLocal.z = glm::dot(dirWS, _w);
  _longitude = std::atan2(dirLocal.x, dirLocal.y);
  _latitude  = std::asin(dirLocal.z);
}

void TrackballController::reinitOrientationFromCamera() {
  _camera->getUVW(_u, _v, _w);
  _u = glm::normalize(_u);
  _v = glm::normalize(_v);
  _w = glm::normalize(_w);
  std::swap(_v, _w);
  _latitude                = 0.0f;
  _longitude               = 0.0f;
  _cameraEyeLookAtDistance = glm::length(_camera->lookAt() - _camera->eye());
}