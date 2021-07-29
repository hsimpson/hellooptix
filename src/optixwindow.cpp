#include "optixwindow.h"
#include <chrono>
#include <iostream>
#include <format>

OptixWindow::OptixWindow(const std::string &title,
                         const Scene &      scene,
                         uint32_t           width,
                         uint32_t           height)
    : GLFWindow(title, width, height) {
  _optixManager = std::make_unique<OptixManager>(scene, width, height);
  _optixManager->setCamera(scene.camera);
}

OptixWindow::~OptixWindow() {
  //_optixManager->writeImage("./image.ppm");
}

void OptixWindow::resize(uint32_t width, uint32_t height) {
  _optixManager->resize(width, height);
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

  // std::cout << std::format("Optix duration: {}ms\n", duration);
}

void OptixWindow::dolly(float offset) {
  _optixManager->dolly(offset);
}

void OptixWindow::move(float offsetX, float offsetY) {
  _optixManager->move(offsetX, offsetY);
}

void OptixWindow::moveLookAt(float offsetX, float offsetY) {
  _optixManager->moveLookAt(offsetX, offsetY);
}

void OptixWindow::rotate(float pitch, float yaw, float roll) {
  _optixManager->rotate(pitch, yaw, roll);
}