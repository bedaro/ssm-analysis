/*
 * This component acts as a shim to allow the NetCDF C++ bindings to work
 * with files in parallel. The old way to open a file in parallel, using
 * a particular flag, has been deprecated in favor of specific C functions
 * nc_open_par and nc_create_par. To implement this, NcFile is subclassed
 * and the open method is overridden to handle passing the MPI objects to
 * these calls.
 *
 * By Ben Roberts, based on code from ncFile.h/ncFile.cpp in the NetCDF
 * C++ library version 4.3.1.
 */
#include "parNcFile.h"
#include "ncCheck.h"

using namespace netCDF;

ParNcFile::ParNcFile(MPI::Comm& comm, MPI::Info& info, const std::string& filePath, const FileMode fMode) {
  open(comm, info, filePath, fMode);
}

void ParNcFile::open(MPI::Comm& comm, MPI::Info& info, const std::string& filePath, const FileMode fMode) {
  if(!nullObject)
    close();

  switch (fMode) {
    case NcFile::write:
      ncCheck(nc_open_par(filePath.c_str(), NC_WRITE, comm, info, &myId),__FILE__,__LINE__);
      break;
    case NcFile::read:
      ncCheck(nc_open_par(filePath.c_str(), NC_NOWRITE, comm, info, &myId),__FILE__,__LINE__);
      break;
    case NcFile::newFile:
      ncCheck(nc_create_par(filePath.c_str(), NC_NETCDF4 | NC_NOCLOBBER, comm, info, &myId),__FILE__,__LINE__);
      break;
    case NcFile::replace:
      ncCheck(nc_create_par(filePath.c_str(), NC_NETCDF4 | NC_CLOBBER, comm, info, &myId),__FILE__,__LINE__);
      break;
    }

  nullObject = false;
}
