#pragma once

#include "glfwindow.h"
#include "optixmanager.h"
#include <memory>
#include "geometry/trianglemesh.h"
#include "cameraController.h"

class OptixWindow : public GLFWindow {
 public:
  OptixWindow(const std::string&                title,
              std::shared_ptr<Scene>            scene,
              std::shared_ptr<CameraController> cameraController,
              uint32_t                          width,
              uint32_t                          height);
  virtual ~OptixWindow();

  virtual void draw();
  virtual void render();

 private:
  static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
  static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
  static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
  static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
  static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

  std::shared_ptr<CameraController> _cameraController;
  std::unique_ptr<OptixManager>     _optixManager;
  uint32_t                          _width;
  uint32_t                          _height;
  GLuint                            _texture{0};
  int                               _mouseButton = -1;
};