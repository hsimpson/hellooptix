#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

class Camera {
 public:
  Camera();
  Camera(glm::vec3 eye, glm::vec3 lookAt, glm::quat rotation, float focalLength, glm::vec2 size);

  const glm::vec3 &eye() const { return _eye; }
  void             setEye(const glm::vec3 &eye);

  const glm::vec3 &lookAt() const { return _lookAt; }
  void             setLookAt(const glm::vec3 &lookAt);

  const glm::vec3 &up() const { return _up; }
  void             setUp(const glm::vec3 &up);

  const glm::quat &rotation() const { return _rotation; }
  void             setRotation(const glm::quat &rotation);

  const glm::vec2 &size() const { return _size; }
  void             setSize(const glm::vec2 &size);

  const float fovY() const { return _fovY; }  // vertical fov in degrees
  void        setFovY(float fovY);            // vertical fov in degrees

  const float focalLength() const { return _focalLength; }
  void        setFocalLength(float focalLength);

  void getUVW(glm::vec3 &U, glm::vec3 &V, glm::vec3 &W) const;

  const bool needsUpdate() const { return _needsUpdate; }
  void       setNeedsUpdate(bool needsUpdate) { _needsUpdate = needsUpdate; }

 private:
  void calcFov();
  void calcVectors();

  glm::vec3   _eye          = {0.0f, 0.0f, 0.0f};          // camera position
  glm::vec3   _lookAt       = {0.0f, 0.0f, 0.0f};          // look at point
  glm::vec3   _up           = {0.0f, 1.0f, 0.0f};          // camera up vector
  glm::quat   _rotation     = glm::identity<glm::quat>();  // camera rotation
  float       _focalLength  = 50.0f;                       // camera focal length (mm)
  float       _aspectRatio  = 1.0;                         // camera aspect ratio
  float       _fovY         = 0.0f;                        // camera field of view vertical in degrees
  float       _fovX         = 0.0f;                        // camera field of view vertical in degrees
  glm::vec2   _size         = {0.0f, 0.0f};                // camera image size
  const float _sensorWidth  = 36.0f;                       // camera sensor width (mm)
  const float _sensorHeight = 24.0f;                       // camera sensor height (mm)
  bool        _needsUpdate  = true;
};
