This C++ program converts a set of ssm_history text files into a single NetCDF
file with a format intended to match the one the SSM can create, although not
every variable from the history files is copied over.

To compile: the NetCDF C++ library is required along with
libboost_program_options. Type `make`.

The program can be built with parallel support using the OpenMPI C++ API.
The NetCDF library will also need to be built with parallel support. To get
this, type `PARALLEL=1 make`. My measurements show speedups of about 3 with 16
cores, so it's not much but it can be helpful.

To test: the included test.py script will run ssmhist2cdf. It requires that
you first run the water quality model for some length of time (1-3 days is
fine) so it produces both ssm_history and NetCDF output at close to matching
time intervals. Then, pass glob patterns of the ssm_history and NetCDF files,
taking care to quote/escape them to prevent them from being interpreted by the
shell.

Example: `./test.py ~/wqmodels/ssm/wqm-test/outputs/{'ssm_history_000*.out',ssm_FVCOMICM_00001.nc}`
