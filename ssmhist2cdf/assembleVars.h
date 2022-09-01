#ifndef _ASSEMBLEVARS_H
#define _ASSEMBLEVARS_H

#include <map>
#include <netcdf>
#include <yaml-cpp/yaml.h>

/**
 * Builds a map of NetCDF output variables keyed by their position in
 * the history file
 */
std::map<size_t, netCDF::NcVar> assembleVars(netCDF::NcFile& ncFile,
    const YAML::Node& config);

#endif
