This C++ program converts a set of ssm_history text files into a single NetCDF
file with a format intended to match the one the SSM can create, although not
every variable from the history files is copied over.

To compile: the NetCDF C++ library is required along with
libboost_program_options and [libyaml-cpp](https://github.com/jbeder/yaml-cpp).
Other Boost headers are also required (spirit, test). Type `make`. Unit tests
will be run automatically.

The program can be built with parallel support using the OpenMPI C++ API.
The NetCDF library will also need to be built with parallel support. To get
this, type `PARALLEL=1 make`. My measurements show speedups of about 3 with 16
cores, so it's not much but it can be helpful. This speedup is likely dependent
upon your system's storage configuration; I ran this test locally on a system
with three ZFS striped vdevs; this may explain the max speedup I encountered.

```
Usage: ssmhist2cdf [options] output-file input-file [input-file ...]
Allowed options:
  -h [ --help ]          produce help message
  -c [ --config ] arg    Specify a config file with the output variables
  -m [ --hyd-model ] arg Specify a hydrodynamic output file to populate grid 
                         from
  -v [ --verbose ]       Verbose output
  --output-file arg      Output netcdf file name
  --input-file arg       Input SSM history file(s) in layer order
```

The program uses some sane defaults on which fields to extract, but these
can be customized by using a YAML config file passed to the program with the
-c option. The included `history.yml` is a sample of the required format.

More recent efforts have been made to make the NetCDF output close to FVCOM
NetCDF output, including by copying the model grid data. You must run the
program with a path to a hydrodynamic output file to get this extra grid data.

Integration test: the included test.py script will run ssmhist2cdf. It
requires that you first run the FVCOM-ICM4 water quality model for some length
of time (1-3 days is fine) so it produces both ssm_history and NetCDF output
at close to matching time intervals. Then, pass glob patterns of the
ssm_history and NetCDF files, taking care to quote/escape them to prevent them
from being interpreted by the shell.

Example: `./test.py ~/wqmodels/ssm/wqm-test/outputs/{'ssm_history_000*.out',ssm_FVCOMICM_00001.nc}`
