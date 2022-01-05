#include <iostream>
#include <fstream>
#include <netcdf>
#include <vector>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>

#define STATEVARS 42

// depth is at index 3
// DO is at index 4
// lDOC is at index 5
// Alg1 is at index 6
// Alg2 is at index 7
// NH4 is at index 8
// NO3 is at index 9
// PO4 is at index 10
// Temp is at index 12
// Salt is at index 13
// rDOC is at index 15
// lPOC is at index 16
// rPOC is at index 17
// DIC is at index 20
// TALK is at index 21
// pH is at index 22
// pCO2 is at index 23

static const std::array<int, 17> output_indices({
  3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 20, 21, 22, 23
});
static const std::array<std::string, 17> output_variables({
  "depth", "do", "ldoc", "alg1", "alg2", "nh4", "no3", "po4", "temp",
  "salt", "rdoc", "lpoc", "rpoc", "dic", "talk", "ph", "pco2"
});

namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
namespace phoenix = boost::phoenix;

int main(int argc, char *argv[]) {
  if(argc < 2) {
    std::cerr << "Usage: ssmhist2cdf input1 [input2 ...] output.cdf" << std::endl;
    return 1;
  }
  try {
    // Initialize the netCDF file
    netCDF::NcFile ncFile(argv[argc-1], netCDF::NcFile::newFile);
    netCDF::NcDim timeDim = ncFile.addDim("time"),
                  sigmaDim = ncFile.addDim("sigma", argc - 2),
                  cellDim; // initialized later once we know how large
    std::vector<netCDF::NcDim> allDims(3);
    std::vector<netCDF::NcVar> allVars(output_variables.size());
    netCDF::NcVar timeVar = ncFile.addVar("times", netCDF::NcType::nc_FLOAT, timeDim);
    allDims[0] = timeDim;
    allDims[1] = sigmaDim;
    float time;
    for(size_t i = 1; i < (size_t)(argc - 1); ++i) {
      try {
        std::ifstream level;
        level.exceptions(std::ifstream::failbit|std::ifstream::badbit);
        level.open(argv[i]);
        size_t cells;
        size_t t = 0;
        std::string line;
        while(! level.eof()) {
          // Get a header line
          getline(level, line);
          qi::phrase_parse(line.begin(), line.end(), (
                qi::float_[phoenix::ref(time) = qi::_1] >
                qi::float_ >
                qi::float_[phoenix::ref(cells) = qi::_1]),
              ascii::space);
          // "cells" tells how many cells there are output variables for.
          // Expect to find STATEVARS times cells values to parse

          std::cout << "TIME " << time << std::endl;
          timeVar.putVar({t}, time);
          if((i == 1) && (t == 0)) {
            // For the first pass of the first file, we need to set
            // up some more CDF variables now that we know the number of
            // cells
            cellDim = ncFile.addDim("cell", cells);
            allDims[2] = cellDim;
            for(size_t j = 0; j < output_variables.size(); ++j) {
              allVars[j] = ncFile.addVar(output_variables[j], netCDF::NcType::nc_FLOAT, allDims);
            }
          }
          size_t total = cells * STATEVARS;
          std::vector<float> data;
          while(data.size() < total) {
            getline(level, line);
            qi::phrase_parse(line.begin(), line.end(),
                *(qi::float_[phoenix::push_back(phoenix::ref(data), qi::_1)]),
                ascii::space);
          }
          for(size_t j = 0; j < output_indices.size(); ++j) {
            netCDF::NcVar v = allVars[j];
            size_t data_index = output_indices[j];
            for(size_t k = 0; k < cells; ++k) {
              v.putVar({t, i - 1, k}, data[data_index * cells + k]);
            }
          }
          ++t;
        }
      } catch(std::ifstream::failure&) {
        std::cout << "Complete parsing for " << argv[i] << " failed at time " << time << std::endl;
      }
    }
  } catch(netCDF::exceptions::NcException& e) {
    std::cout << e.what() << std::endl;
    return 1;
  }
  return 0;
}
