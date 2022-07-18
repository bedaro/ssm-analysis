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
