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
#ifndef ParNcFileClass
#define ParNcFileClass

#include <ncFile.h>
#include <netcdf_par.h>
#include <mpi.h>

namespace netCDF {

  class ParNcFile : public NcFile {
    public:
      ParNcFile(MPI::Comm& comm, MPI::Info& info, const std::string& filePath, const FileMode fMode);

      void open(MPI::Comm& comm, MPI::Info& info, const std::string& filePath, const FileMode fMode);
      // create, etc is not implemented
  };
}

#endif
