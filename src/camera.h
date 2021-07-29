#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

class Camera {
 public:
  Camera(float fovY, float zNear, float zFar);

  void setRotation(const glm::quat &rotationQuaternion);
  void setRotation(const glm::mat4 &rotationMatrix);
  void setPosition(const glm::vec3 &position);

  void translate(const glm::vec3 &translation);
  void rotate(float pitch, float yaw, float roll);

  const glm::vec3 &position() const {
    return _position;
  }
  const glm::vec3 &right() const {
    return _right;
  }
  const glm::vec3 &up() const {
    return _up;
  }
  const glm::vec3 &front() const {
    return _front;
  }
  const float fovY() const {
    return _fovY;
  }
  const float zNear() const {
    return _zNear;
  }
  const float zFar() const {
    return _zFar;
  }

 private:
  void updateVectors();

  float _fovY  = glm::radians(25.0f);  // vertical field of view in radians
  float _zNear = 0.1f;
  float _zFar  = 1000.0f;

  glm::quat _rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);  // camera matrix

  glm::vec3 _position = {0.0f, 0.0f, 0.0f};  // camera position
  glm::vec3 _right    = {1.0f, 0.0f, 0.0f};  // camera right vector
  glm::vec3 _up       = {0.0f, 1.0f, 0.0f};  // camera up vector
  glm::vec3 _front    = {0.0f, 0.0f, 1.0f};  // camera front vector

  constexpr static glm::vec4 RIGHT = {1.0f, 0.0f, 0.0f, 1.0f};
  constexpr static glm::vec4 UP    = {0.0f, 1.0f, 0.0f, 1.0f};
  constexpr static glm::vec4 FRONT = {0.0f, 0.0f, 1.0f, 1.0f};
};
