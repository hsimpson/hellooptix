#include "glfwindow.h"
#include "optixmanager.h"
#include <memory>

class OptixWindow : public GLFWindow {
 public:
  OptixWindow(const std::string& title, uint32_t width, uint32_t height);
  virtual ~OptixWindow();

  virtual void resize(uint32_t width, uint32_t height);
  virtual void draw();
  virtual void render();

 private:
  std::unique_ptr<OptixManager> _optixManager;

  GLuint _texture{0};
};