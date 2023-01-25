#include "../copyFromHyd.h"
#define BOOST_TEST_MODULE copyFromHyd
#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <random>
#include "ncCheck.h"

// Define an in-memory NcFile class similar to ParNcFile
namespace netCDF {
  class InMemNcFile : public NcFile {
    public:
      InMemNcFile() {
        if(!nullObject)
          close();
        
        std::string filePath = "/dummy/path.nc";

        ncCheck(nc_create(filePath.c_str(), NC_DISKLESS, &myId),__FILE__,__LINE__);

        nullObject = false;
      }
  };
}

BOOST_AUTO_TEST_CASE( copyFromHyd_test )
{
  // Mock NetCDF files
  netCDF::InMemNcFile f1 = netCDF::InMemNcFile(),
    f2 = netCDF::InMemNcFile();
  
  // Populate file 1 with some basic data
  netCDF::NcDim d1 = f1.addDim("d1", 23);
  netCDF::NcDim d2 = f1.addDim("d2", 19);
  std::vector<netCDF::NcDim> dims = {d1, d2};
  netCDF::NcVar v1 = f1.addVar("v1", netCDF::NcType::nc_FLOAT, d1);
  v1.putAtt("a1", "a1 val");
  v1.putAtt("a2", "a2 val");
  float data1[23];
  std::mt19937 rng;
  std::normal_distribution<float> normal_dist(0, 1);
  for(size_t i = 0; i < 23; ++i) {
    data1[i] = normal_dist(rng);
  }
  v1.putVar(data1);
  netCDF::NcVar v2 = f1.addVar("v2", netCDF::NcType::nc_FLOAT, dims);
  v2.putAtt("a3", "a3 val");
  v2.putAtt("a4", "a4 val");
  float data2[23][19];
  for(size_t i = 0; i < 23; ++i) {
    for(size_t j = 0; j < 19; ++j) {
      data2[i][j] = normal_dist(rng);
    }
  }
  v2.putVar(data2);

  // Copy the dimensions to the second file
  //netCDF::NcDim d1_2 = f2.addDim("d1", 23);
  //netCDF::NcDim d2_2 = f2.addDim("d2", 19);

  // Test
  copyVariable(v1, f2);
  netCDF::NcVar v1_2 = f2.getVar("v1");
  BOOST_CHECK_EQUAL(v1_2.getType().getName(), v1.getType().getName());
  BOOST_CHECK_EQUAL(v1_2.getDimCount(), v1.getDimCount());
  BOOST_CHECK_EQUAL(v1_2.getAttCount(), v1.getAttCount());
  std::string av1, av1_2;
  v1_2.getAtt("a1").getValues(av1_2);
  v1.getAtt("a1").getValues(av1);
  BOOST_CHECK_EQUAL(av1_2, av1);
  float value1, value2;
  v1_2.getVar({ 10 }, &value2);
  v1.getVar({ 10 }, &value1);
  BOOST_CHECK_EQUAL(value2, value1);

  f1.close();
  f2.close();
}

// vim: set shiftwidth=2 expandtab:
