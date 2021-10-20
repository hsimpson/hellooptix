#include "trackballController.h"
#include "spdlog/spdlog.h"

TrackballController::TrackballController(std::shared_ptr<Camera> camera) : CameraController(camera) {
  _cameraEyeLookAtDistance = glm::length(_camera->lookAt() - _camera->eye());
}

TrackballController::~TrackballController() {
}

void TrackballController::zoom(float delta) {
  float zoom = delta > 0.0f ? 1.0f / _zoomFactor : _zoomFactor;
  _cameraEyeLookAtDistance *= zoom;
  glm::vec3 eye    = _camera->eye();
  glm::vec3 lookAt = _camera->lookAt();
  _camera->setEye(lookAt + (eye - lookAt) * zoom);
}

void TrackballController::resize(uint32_t width, uint32_t height) {
  _camera->setSize({static_cast<float>(width),
                    static_cast<float>(height)});
}

void TrackballController::rotate(const glm::ivec2& mousePosition) {
  glm::ivec2 mouseDelta = mousePosition - _mousePosition;
  _mousePosition        = mousePosition;
  spdlog::debug("TrackballController::rotate: {},{}", mouseDelta.x, mouseDelta.y);

  glm::vec3 moveDirection{mouseDelta.x, mouseDelta.y, 0.0f};
  _angle = glm::length(moveDirection);

  glm::vec3 eye    = _camera->eye();
  glm::vec3 lookAt = _camera->lookAt();

  if (_angle) {
    eye = eye - lookAt;

    glm::vec3 eyeDirection = glm::normalize(eye);
    // glm::vec3 upDirection   = glm::normalize(_cameraUp);
    glm::vec3 upDirection   = {0.0f, 1.0f, 0.0f};
    glm::vec3 sideDirection = glm::normalize(glm::cross(upDirection, eyeDirection));

    upDirection   = upDirection * static_cast<float>(-mouseDelta.y);
    sideDirection = sideDirection * static_cast<float>(mouseDelta.x);
    moveDirection = upDirection + sideDirection;
    _axis         = glm::normalize(glm::cross(moveDirection, eye));

    _angle *= _rotationFactor;

    glm::quat quaternion = glm::angleAxis(_angle, _axis);

    eye          = glm::rotate(quaternion, eye);
    _cameraRight = glm::rotate(quaternion, _cameraRight);
    _cameraUp    = glm::rotate(quaternion, _cameraUp);

    _camera->setEye(eye);
  } else {
    spdlog::debug("TrackballController::rotate: angle is zero");

    eye                  = eye - lookAt;
    glm::quat quaternion = glm::angleAxis(_angle, _axis);
    eye                  = glm::rotate(quaternion, eye);
    _cameraRight         = glm::rotate(quaternion, _cameraRight);
    _cameraUp            = glm::rotate(quaternion, _cameraUp);
  }

  // _camera->setEye(eye);
}

void TrackballController::pan(const glm::ivec2& mousePosition) {
  glm::ivec2 mouseDelta = mousePosition - _mousePosition;
  _mousePosition        = mousePosition;

  glm::vec3 eye    = _camera->eye();
  glm::vec3 lookAt = _camera->lookAt();

  eye -= _cameraRight * static_cast<float>(mouseDelta.x) * _panFactor;
  lookAt -= _cameraRight * static_cast<float>(mouseDelta.x) * _panFactor;

  eye += _cameraUp * static_cast<float>(mouseDelta.y) * _panFactor;
  lookAt += _cameraUp * static_cast<float>(mouseDelta.y) * _panFactor;

  _camera->setEye(eye);
  _camera->setLookAt(lookAt);

  spdlog::debug("TrackballController::pan: {},{}", mouseDelta.x, mouseDelta.y);
}
