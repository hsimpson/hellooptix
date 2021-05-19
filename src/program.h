#pragma once

#include <string>

class Program {
 public:
  Program(std::string filename);

  const std::string getPTX();

 private:
  std::string _filename;
};
