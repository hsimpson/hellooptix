#include "glfwindow.h"
#include <iostream>
#include "spdlog/spdlog.h"

static void glfw_error_callback(int error, const char* description) {
  spdlog::error("GLFW Error {}: {}", error, description);
}

GLFWindow::GLFWindow(const std::string& title, uint32_t width, uint32_t height) {
  glfwSetErrorCallback(glfw_error_callback);

  if (!glfwInit())
    exit(EXIT_FAILURE);

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

  _windowHandle = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
  if (!_windowHandle) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  glfwSetWindowUserPointer(_windowHandle, this);
  glfwMakeContextCurrent(_windowHandle);
  glfwSwapInterval(1);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }
}

GLFWindow::~GLFWindow() {
  glfwDestroyWindow(_windowHandle);
  glfwTerminate();
}

void GLFWindow::run() {
  while (!glfwWindowShouldClose(_windowHandle)) {
    render();
    draw();

    glfwSwapBuffers(_windowHandle);
    glfwPollEvents();
  }
}
