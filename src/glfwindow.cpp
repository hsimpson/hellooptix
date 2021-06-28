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
}

GLFWindow::~GLFWindow() {
  glfwDestroyWindow(_handle);
  glfwTerminate();
}

void GLFWindow::run() {
  glfwSetFramebufferSizeCallback(_handle, GLFWindow::framebufferResizeCallback);

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
