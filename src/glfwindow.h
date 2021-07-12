#pragma once

// #define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <string>

class GLFWindow {
 public:
  GLFWindow(const std::string& title, uint32_t width, uint32_t height);
  virtual ~GLFWindow();

  void         run();
  virtual void resize(uint32_t width, uint32_t height) = 0;
  virtual void draw()                                  = 0;
  virtual void render()                                = 0;

 protected:
  uint32_t _width;
  uint32_t _height;

 private:
  // glfw call backs
  static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

  GLFWwindow* _handle{nullptr};
};