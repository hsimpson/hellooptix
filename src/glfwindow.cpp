#include "glfwindow.h"
#include <iostream>

static void glfw_error_callback(int error, const char* description) {
  std::cerr << "Error: " << description << std::endl;
}

GLFWindow::GLFWindow(const std::string& title, uint32_t width, uint32_t height)
    : _width(width), _height(height) {
  glfwSetErrorCallback(glfw_error_callback);

  if (!glfwInit())
    exit(EXIT_FAILURE);

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

  _handle = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
  if (!_handle) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  glfwSetWindowUserPointer(_handle, this);
  glfwMakeContextCurrent(_handle);
  glfwSwapInterval(1);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }
  glfwSetFramebufferSizeCallback(_handle, GLFWindow::framebufferResizeCallback);
  glfwSetKeyCallback(_handle, GLFWindow::keyCallback);
  glfwSetScrollCallback(_handle, GLFWindow::scrollCallback);
}

GLFWindow::~GLFWindow() {
  glfwDestroyWindow(_handle);
  glfwTerminate();
}

void GLFWindow::run() {
  while (!glfwWindowShouldClose(_handle)) {
    render();
    draw();

    glfwSwapBuffers(_handle);
    glfwPollEvents();
  }
}

void GLFWindow::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
  auto w     = static_cast<GLFWindow*>(glfwGetWindowUserPointer(window));
  w->_width  = width;
  w->_height = height;
  w->resize(width, height);
}

void GLFWindow::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto w = static_cast<GLFWindow*>(glfwGetWindowUserPointer(window));

  float offsetX = 0.0f;
  float offsetY = 0.0f;

  if (mods == GLFW_MOD_ALT) {
    switch (key) {
      case GLFW_KEY_W:
        offsetY = 1.0f;
        break;
      case GLFW_KEY_S:
        offsetY = -1.0f;
        break;
      case GLFW_KEY_A:
        offsetX = -1.0f;
        break;
      case GLFW_KEY_D:
        offsetX = 1.0f;
        break;
    }
    w->moveLookAt(offsetX, offsetY);
  } else {
    switch (key) {
      case GLFW_KEY_W:
        offsetY = 1.0f;
        break;
      case GLFW_KEY_S:
        offsetY = -1.0f;
        break;
      case GLFW_KEY_A:
        offsetX = -1.0f;
        break;
      case GLFW_KEY_D:
        offsetX = 1.0f;
        break;
    }
    w->move(offsetX, offsetY);
  }
}

void GLFWindow::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
  auto w = static_cast<GLFWindow*>(glfwGetWindowUserPointer(window));
  w->zoom(yoffset);
}
