//============================================================================
// Description: ConfigReader class
// Makoto Powers
//
// This class provides utility functions for reading in configuration files.
//============================================================================

#pragma once

//============================================================================
// INCLUDES
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

//============================================================================

namespace Tools {

class ConfigReader {
 private:
 public:
  std::vector<std::string> vec;

  void readConfigFile(std::string fname);

  std::string selectConfigEntry(std::string entr) const;
  template <typename returnValue>
  returnValue returnConfigValue(std::string entr) const {
    std::string str = selectConfigEntry(entr);
    std::stringstream ss(str);
    returnValue convertedValue;
    if (ss >> convertedValue) {
      return convertedValue;
    } else {
      throw std::runtime_error("Error: Could not convert config value to desired type.");
    }
  }
};

}  // namespace Tools