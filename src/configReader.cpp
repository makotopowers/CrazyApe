//============================================================================
// Implementation of the ConfigReader class.
// Makoto Powers
//============================================================================

//============================================================================
// INCLUDES
#include "configReader.hpp"

//============================================================================

namespace Tools {

void ConfigReader::readConfigFile(std::string fname) {
  try {
    vec.clear();
    std::ifstream inFile;
    inFile.open(fname);

    std::string str;
    while (getline(inFile, str)) {
      vec.push_back(str);
    }
    inFile.close();
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
  }
}

std::string ConfigReader::selectConfigEntry(std::string entr) const {
  std::string sub;
  for (int i = 0; i < (int)vec.size(); i++) {
    sub = vec[i].substr(0, entr.size());
    if (sub == entr) {
      return vec[i].substr(entr.size() + 1, vec[i].size());
    }
  }
  return "0";
}

}  // namespace Tools