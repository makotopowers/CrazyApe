//============================================================================
// Implementation of the Utilities class.
// Makoto Powers
//============================================================================

//============================================================================
// INCLUDES
#include "utilities.hpp"

//============================================================================

namespace Tools {

int Utilities::debug = 0;

Utilities::Utilities() {}

Utilities::~Utilities() {}

void Utilities::Debug(std::string message) {
  if (Utilities::debug > 1) {
    std::cout << "[[DEBUG]] ";
    std::cout << message << std::endl;
  }
}

void Utilities::Log(std::string message) {
  if (Utilities::debug > 0) {
    std::cout << "[[LOG]] ";
    std::cout << message << std::endl;
  }
}

void Utilities::setDebug(int debug) {
  this->debug = debug;
}

}  // namespace Tools