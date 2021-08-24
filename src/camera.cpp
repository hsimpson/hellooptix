#include "./camera.h"

Camera::Camera(float fovY, float zNear, float zFar)
    : _fovY(fovY), _zNear(zNear), _zFar(zFar) {
}

void Camera::setRotation(const glm::quat& rotationQuaternion) {
  _rotation = rotationQuaternion;
  updateVectors();
}

void Camera::setRotation(const glm::mat4& rotationMatrix) {
  _rotation = glm::toQuat(rotationMatrix);
  updateVectors();
}

void Camera::setPosition(const glm::vec3& position) {
  _position = position;
  updateVectors();
}

void Camera::translate(const glm::vec3& translation) {
  _position += translation;
  // updateVectors();
}

void Camera::rotate(float pitch, float yaw, float roll) {
  _rotation = glm::rotate(_rotation, pitch, glm::vec3(1.0f, 0.0f, 0.0f));
  _rotation = glm::rotate(_rotation, yaw, glm::vec3(0.0f, 1.0f, 0.0f));
  _rotation = glm::rotate(_rotation, roll, glm::vec3(0.0f, 0.0f, 1.0f));
  updateVectors();
}

void Camera::updateVectors() {
  glm::mat4 rotationMatrix = glm::toMat4(_rotation);
  _right                   = glm::normalize(glm::vec3(rotationMatrix * Camera::RIGHT));
  _up                      = glm::normalize(glm::vec3(rotationMatrix * Camera::UP));
  _forward                 = glm::normalize(glm::vec3(rotationMatrix * Camera::FORWARD));
}