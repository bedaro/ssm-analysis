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
