#include "optixwindow.h"
#include <chrono>
#include <iostream>
#include <format>
#include "spdlog/spdlog.h"

OptixWindow::OptixWindow(const std::string&                title,
                         std::shared_ptr<Scene>            scene,
                         std::shared_ptr<CameraController> cameraController,
                         uint32_t                          width,
                         uint32_t                          height)
    : GLFWindow(title, width, height),
      _width(width),
      _height(height),
      _cameraController(cameraController) {
  _optixManager = std::make_unique<OptixManager>(scene, width, height);

  glfwSetFramebufferSizeCallback(_windowHandle, OptixWindow::framebufferResizeCallback);
  glfwSetScrollCallback(_windowHandle, OptixWindow::scrollCallback);
  glfwSetCursorPosCallback(_windowHandle, OptixWindow::cursorPosCallback);
  glfwSetMouseButtonCallback(_windowHandle, OptixWindow::mouseButtonCallback);
  glfwSetKeyCallback(_windowHandle, OptixWindow::keyCallback);
}

OptixWindow::~OptixWindow() {
  // _optixManager->writeImage("./image.ppm");
}

void OptixWindow::draw() {
  if (_texture == 0)
    glGenTextures(1, &_texture);

  glBindTexture(GL_TEXTURE_2D, _texture);
  GLenum texFormat = GL_RGBA;
  GLenum texelType = GL_UNSIGNED_BYTE;
  glTexImage2D(GL_TEXTURE_2D, 0, texFormat, _width, _height, 0, GL_RGBA,
               texelType, _optixManager->getOutputBuffer()->getHostPointer());

  glDisable(GL_LIGHTING);
  glColor3f(1, 1, 1);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, _texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glDisable(GL_DEPTH_TEST);

  glViewport(0, 0, _width, _height);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.f, (float)_width, 0.f, (float)_height, -1.f, 1.f);

  glBegin(GL_QUADS);
  {
    glTexCoord2f(0.f, 0.f);
    glVertex3f(0.f, 0.f, 0.f);

    glTexCoord2f(0.f, 1.f);
    glVertex3f(0.f, (float)_height, 0.f);

    glTexCoord2f(1.f, 1.f);
    glVertex3f((float)_width, (float)_height, 0.f);

    glTexCoord2f(1.f, 0.f);
    glVertex3f((float)_width, 0.f, 0.f);
  }
  glEnd();
}

void OptixWindow::render() {
  auto t1 = std::chrono::high_resolution_clock::now();
  _optixManager->launch();
  auto t2       = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double, std::chrono::milliseconds::period>(t2 - t1).count();

  spdlog::debug("Optix duration: {:.3f} ms", duration);
}

void OptixWindow::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
  auto w     = static_cast<OptixWindow*>(glfwGetWindowUserPointer(window));
  w->_width  = width;
  w->_height = height;
  w->_cameraController->resize(width, height);
  w->_optixManager->resize(width, height);
}

void OptixWindow::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
  auto w = static_cast<OptixWindow*>(glfwGetWindowUserPointer(window));
  w->_cameraController->zoom(static_cast<float>(yoffset));
}

void OptixWindow::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
  auto w = static_cast<OptixWindow*>(glfwGetWindowUserPointer(window));

  if (w->_mouseButton == GLFW_MOUSE_BUTTON_LEFT) {
    w->_cameraController->setViewMode(CameraController::LookAtFixed);
    w->_cameraController->updateTracking({static_cast<int32_t>(xpos), static_cast<int32_t>(ypos)});
  } else if (w->_mouseButton == GLFW_MOUSE_BUTTON_RIGHT) {
    w->_cameraController->setViewMode(CameraController::EyeFixed);
    w->_cameraController->updateTracking({static_cast<int32_t>(xpos), static_cast<int32_t>(ypos)});
  }
}

void OptixWindow::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  auto w = static_cast<OptixWindow*>(glfwGetWindowUserPointer(window));

  double xpos, ypos;
  glfwGetCursorPos(window, &xpos, &ypos);

  if (action == GLFW_PRESS) {
    w->_mouseButton = button;
    w->_cameraController->startTracking({static_cast<int32_t>(xpos), static_cast<int32_t>(ypos)});
  } else {
    w->_mouseButton = -1;
  }
}

void OptixWindow::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto w = static_cast<OptixWindow*>(glfwGetWindowUserPointer(window));
}