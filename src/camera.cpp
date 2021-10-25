#include "camera.h"

Camera::Camera() {
  calcFov();
}

Camera::Camera(glm::vec3 eye, glm::vec3 lookAt, glm::quat rotation, float focalLength, glm::vec2 size)
    : _eye(eye),
      _lookAt(lookAt),
      _rotation(rotation),
      _focalLength(focalLength),
      _size(size),
      _aspectRatio(size.x / size.y) {
  calcFov();
}

void Camera::calcFov() {
  const float horizontalFov = 2.0f * std::atan(0.5f * _sensorWidth / _focalLength);
  const float verticalFov   = 2.0f * std::atan(0.5f * _sensorHeight / _focalLength);

  /*
  hints:
  - http://paulbourke.net/miscellaneous/lens/
  - https://www.panavision.com/sites/default/files/docs/documentLibrary/2%20Sensor%20Size%20FOV%20(2).pdf
  - https://www.scantips.com/lights/fieldofview.html#top
  */

  if (_aspectRatio > 1.0f) {
    _fovX = glm::degrees(horizontalFov);
    _fovY = glm::degrees(2.0f * std::atan(_size.y * std::tan(horizontalFov * 0.5f) / _size.x));
  } else {
    // ToDo: check Portrait mode
    _fovX = glm::degrees(2.0f * std::atan(_size.x * std::tan(verticalFov * 0.5f) / _size.y));
    _fovY = glm::degrees(verticalFov);
  }
}

void Camera::setFovY(float fovY) {
  _needsUpdate     = true;
  _fovY            = fovY;
  const float fovy = glm::radians(_fovY);

  if (_aspectRatio > 1.0f) {
    float horizontalFov = std::atan((std::tan(fovy / 2.0f) * _size.x) / _size.y) / 0.5f;
    _fovX               = glm::degrees(horizontalFov);
    _focalLength        = 1.0f / ((std::tan(horizontalFov / 2.0f) / 0.5f) / _sensorWidth);
  } else {
    // ToDo: check Portrait mode
    float verticalFov = fovy;
    _fovX             = std::atan((std::tan(fovy / 2.0f) * _size.x) / _size.y) / 0.5f;
    _focalLength      = 1.0f / ((std::tan(verticalFov / 2.0f) / 0.5f) / _sensorHeight);
  }
}

void Camera::getUVW(glm::vec3 &U, glm::vec3 &V, glm::vec3 &W) const {
  W          = _lookAt - _eye;
  float wlen = glm::length(W);
  U          = glm::normalize(glm::cross(W, _up));
  V          = glm::normalize(glm::cross(U, W));
  float vlen = wlen * std::tan(glm::radians(_fovY) / 2.0f);
  V *= vlen;
  float ulen = vlen * _aspectRatio;
  U *= ulen;
}

void Camera::setSize(const glm::vec2 &size) {
  _needsUpdate = true;
  _size        = size;
  _aspectRatio = size.x / size.y;
  calcFov();
}

void Camera::setFocalLength(float focalLength) {
  _needsUpdate = true;
  _focalLength = focalLength;
  calcFov();
}

void Camera::setEye(const glm::vec3 &eye) {
  _needsUpdate = true;
  _eye         = eye;
}

void Camera::setLookAt(const glm::vec3 &lookAt) {
  _needsUpdate = true;
  _lookAt      = lookAt;
}

void Camera::setUp(const glm::vec3 &up) {
  _needsUpdate = true;
  _up          = up;
}

void Camera::setRotation(const glm::quat &rotation) {
  _needsUpdate = true;
  _rotation    = rotation;
}