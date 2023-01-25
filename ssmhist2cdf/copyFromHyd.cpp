#include "copyFromHyd.h"
#include <vector>

using netCDF::NcFile;
using netCDF::NcDim;
using netCDF::NcVar;
using netCDF::NcVarAtt;

void copyVariable(const NcVar& var, NcFile& dest) {
  // Get the variable's dimensions
  std::vector<netCDF::NcDim> dims;
  size_t size = 1;
  for(netCDF::NcDim d: var.getDims()) {
    netCDF::NcDim destdim = dest.getDim(d.getName());
    if(destdim.isNull())
      destdim = dest.addDim(d.getName(), d.getSize());
    dims.push_back(destdim);

    size *= d.getSize();
  }

  // Create the new variable with the same dimensions
  NcVar newVar = dest.addVar(var.getName(), var.getType(), dims);

  // Keep track of the total size of the data for later
  size *= var.getType().getSize();

  // Copy over all the attributes
  std::map<std::string, NcVarAtt> atts = var.getAtts();
  size_t max_size = 1000;
  void *b = malloc(max_size);
  for(auto const& [name, att]: atts) {  // C++17
    size_t attsz = att.getAttLength();
    if(attsz > max_size) {
      free(b);
      b = malloc(attsz);
      max_size = attsz;
    }
    att.getValues(b);
    newVar.putAtt(name, att.getType(), attsz, b);
  }
  free(b);

  // Copy the data
  void *buffer = malloc(size);
  var.getVar(buffer);
  newVar.putVar(buffer);
  free(buffer);
}

void copyFromHyd(const NcFile& ncHyd, NcFile& ncFile) {
  // Copy variable attributes
  copyVariable(ncHyd.getVar("x"), ncFile);
  copyVariable(ncHyd.getVar("y"), ncFile);
  copyVariable(ncHyd.getVar("lat"), ncFile);
  copyVariable(ncHyd.getVar("lon"), ncFile);
  copyVariable(ncHyd.getVar("siglay"), ncFile);
  copyVariable(ncHyd.getVar("siglay_shift"), ncFile);
  copyVariable(ncHyd.getVar("siglev"), ncFile);
  copyVariable(ncHyd.getVar("nv"), ncFile);
}
// vim: set shiftwidth=2 expandtab:
