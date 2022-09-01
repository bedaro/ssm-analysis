#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <netcdf>
#include <yaml-cpp/yaml.h>
#include "assembleVars.h"

std::map<size_t, netCDF::NcVar> assembleVars(netCDF::NcFile& ncFile,
    const YAML::Node& config) {
  // FIXME type checking of integers/booleans is needed but I don't
  // know how to do it
  std::map<size_t, netCDF::NcVar> allVars;

  if(! config["output_indices"] ||
      (config["output_indices"].Type() != YAML::NodeType::Map)) {
    throw "Parse error in config file: output_indices should be a "
      "Map of integers";
  }

  for(YAML::const_iterator it=config["output_indices"].begin();
      it != config["output_indices"].end(); ++it) {
    
    size_t j = it->first.as<size_t>();
    YAML::Node data = it->second;

    if((data.Type() != YAML::NodeType::Map) || ! data["variable"]) {
      throw "Parse error in config file: index " + std::to_string(j) +
        " is missing required subfield 'variable'";
    }
    // Go through the YAML node and compile all
    // attributes, dimensions for the variable
    std::vector<netCDF::NcDim> dims = {ncFile.getDim("time"),
      ncFile.getDim("siglay"), ncFile.getDim("node")};
    std::map<std::string, std::string> atts;
    // Iterate over all the nodes
    for(YAML::const_iterator it2 = data.begin(); it2 != data.end(); ++it2) {
      std::string name = it2->first.as<std::string>();
      if(name == "variable") {
        // We'll use this later
        continue;
      } else if(name == "per_layer") {
        if(! it2->second.as<bool>()) {
          // Remove siglay from the dimensions
          dims.erase(std::remove(dims.begin(), dims.end(),
                ncFile.getDim("siglay")), dims.end());
        }
      } else if(name == "per_time") {
        if(! it2->second.as<bool>()) {
          // Remove time from the dimensions
          dims.erase(std::remove(dims.begin(), dims.end(), 
                ncFile.getDim("time")), dims.end());
        }
      } else {
        // Add this to atts
        atts[name] = it2->second.as<std::string>();
      }
    }
    allVars[j] = ncFile.addVar(data["variable"].as<std::string>(),
                               netCDF::NcType::nc_FLOAT, dims);
    // Assign the attributes
    for(auto& [key, val] : atts) {
      allVars[j].putAtt(key, val);
    }
  }

  return allVars;
}
