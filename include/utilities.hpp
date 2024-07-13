//============================================================================
// Description: Utilities class
// Makoto Powers
//
// This class provides utility functions for logging and debugging.
//============================================================================

#pragma once

//============================================================================
// INCLUDES
#include <iostream>
#include <string>

//============================================================================

namespace Tools {

class Utilities {

  //   @brief Utilities class
  //
  //   This class provides utility functions for logging and debugging.

 public:
  static int debug;

  Utilities();
  ~Utilities();

  static void Log(std::string message);
  static void Debug(std::string message);
  void setDebug(int debug);
};

}  // namespace Tools
