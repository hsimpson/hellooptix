#pragma once

// #define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <string>
#include <glm/glm.hpp>

class GLFWindow {
 public:
  GLFWindow(const std::string& title, uint32_t width, uint32_t height);
  virtual ~GLFWindow();

  void         run();
  virtual void draw()   = 0;
  virtual void render() = 0;

 protected:
  GLFWwindow* _windowHandle;
};
