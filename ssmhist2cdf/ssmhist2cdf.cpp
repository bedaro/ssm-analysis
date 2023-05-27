#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <netcdf>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include "assembleVars.h"
#include "copyFromHyd.h"
#include "historyfile.hpp"
#include "timer.hpp"
#ifdef PARALLEL
  #include <mpi.h>
  #include "parNcFile.h"

  // Check for C++ bindings (OpenMPI, Intel/MPICH)
  #if (OMPI_BUILD_CXX_BINDINGS == 0) || MPICH_SKIP_MPICXX
    #error "OpenMPI C++ bindings are missing, cannot compile"
  #endif

  #define abort(val) MPI::COMM_WORLD.Abort(val == 0? 0: -1)
#else
  #define abort(val) return val
#endif

#define USAGE "Usage: ssmhist2cdf [options] output-file input-file [input-file ...]"

// Default configuration
static const std::string default_variables =
  "output_indices:\n"
  "  0:\n"
  "    variable: 'h'\n"
  "    long_name: 'bathymetry'\n"
  "    units: 'meters'\n"
  "    positive: 'down'\n"
  "    per_layer: false\n"
  "  1:\n"
  "    variable: 'zeta'\n"
  "    long_name: 'Water Surface Elevation'\n"
  "    units: 'meters'\n"
  "    positive: 'up'\n"
  "    per_layer: false\n"
  "  3:\n"
  "    variable: 'depth'\n"
  "    long_name: 'depth'\n"
  "    units: 'meters'\n"
  "  4:\n"
  "    variable: 'DOXG'\n"
  "    long_name: 'dissolved oxygen'\n"
  "    units: 'MG/L'\n"
  "  6:\n"
  "    variable: 'B1'\n"
  "    long_name: 'algal group 1'\n"
  "    units: 'gC meters-3'\n"
  "  7:\n"
  "    variable: 'B2'\n"
  "    long_name: 'algal group 2'\n"
  "    units: 'gC meters-3'\n"
  "  8:\n"
  "    variable: 'NH4'\n"
  "    long_name: 'ammonia'\n"
  "    units: 'gN meters-3'\n"
  "  9:\n"
  "    variable: 'NO3'\n"
  "    long_name: 'nitrate+nitrite'\n"
  "    units: 'gN meters-3'\n"
  "  10:\n"
  "    variable: 'PO4'\n"
  "    long_name: 'phosphate'\n"
  "    units: 'gP meters-3'\n";

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
  int rank = -1;
  float time = 0;
#ifndef PARALLEL
  rank = 0;
  const size_t procs = 1;
#else
  MPI::Init(argc, argv);
  MPI::Comm& mpiComm = MPI::COMM_WORLD;
  rank = mpiComm.Get_rank();
  const size_t procs = mpiComm.Get_size();
#endif

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("config,c", po::value<std::string>(),
        "Specify a config file with the output variables")
    ("hyd-model,m", po::value<std::string>(),
        "Specify a hydrodynamic output file to populate grid from")
    ("verbose,v", "Verbose output")
    ("output-file", po::value<std::string>()->required(),
        "Output netcdf file name")
    ("input-file", po::value< std::vector<std::string> >()->required(),
        "Input SSM history file(s) in layer order")
  ;
  po::positional_options_description p;
  p.add("output-file", 1).add("input-file", -1);

  po::variables_map vm;
  HistoryFile hf;
  try {
    po::store(po::command_line_parser(argc, argv).
              options(desc).positional(p).run(), vm);

    // Fix from https://stackoverflow.com/a/5517755
    if(vm.count("help") && (rank == 0)) {
      std::cout << USAGE << std::endl << desc << std::endl;
      abort(0);
    }

    po::notify(vm);
  } catch (po::error& e) {
    if(rank == 0) {
      std::cerr << USAGE << std::endl
                << "Try ssmhist2cdf --help" << std::endl;
    }
    abort(1);
  }

  YAML::Node config;
  if(vm.count("config")) {
#ifndef NDEBUG
    std::cout << "Reading configuration from " << vm["config"].as<std::string>() << std::endl;
#endif
    config = YAML::LoadFile(vm["config"].as<std::string>());
  } else {
    config = YAML::Load(default_variables);
  }

  const std::vector<std::string> input_files =
      vm["input-file"].as< std::vector<std::string> >();
  const std::string output_file = vm["output-file"].as<std::string>();
  bool verbose = vm.count("verbose");
  // Field width of input file index
  int input_w = std::ceil(std::log10(input_files.size() + 1));

  try {
#ifdef PARALLEL
    verbose &= (rank == 0);
    if(verbose) {
      std::cout << "Running in parallel, " << procs << " processes" << std::endl;
    }

    mpiComm.Set_errhandler(MPI::ERRORS_THROW_EXCEPTIONS);
    MPI::Info mpiInfo = MPI::INFO_NULL;
#endif
    unsigned long nodes = 0, times = 0;
    // Only do this once to ensure processes can't be out of sync due
    // to I/O based race conditions
    if(rank == 0) {
      // Read the first header line of the first file to get node and time
      // count
      try {
        hf.set_file(input_files[0]);
        nodes = hf.get_nodes();
        times = hf.get_times();
      } catch(HistoryFileException& e) {
        std::cerr << "Error in file " << e.getHistoryFile()->get_path() << std::endl;
        std::cerr << e.what() << std::endl;
      }
    }
#ifdef PARALLEL
    // Distribute the history file properties to the other processes
    mpiComm.Bcast(&nodes, 1, MPI::UNSIGNED_LONG, 0);
    mpiComm.Bcast(&times, 1, MPI::UNSIGNED_LONG, 0);
#endif
    // Check if earlier file read succeeded
    if(nodes == 0) {
      abort(1);
    }

    int t_w = std::ceil(std::log10(times + 1));

#ifdef PARALLEL
    netCDF::NcFile::FileMode fMode = netCDF::NcFile::newFile;
#else
    netCDF::NcFile ncFile(output_file, netCDF::NcFile::newFile);
#endif
    if(vm.count("hyd-model")) {
      // Copy the FVCOM data only in the top-rank process, before opening the
      // output file in parallel. This requires changing the open mode
      if(rank == 0) {
        if(verbose) {
          std::cout << "Copying data from hydro model " << vm["hyd-model"].as<std::string>() << std::endl;
        }
        netCDF::NcFile ncHyd(vm["hyd-model"].as<std::string>(), netCDF::NcFile::read);
#ifdef PARALLEL
        netCDF::NcFile f(output_file, netCDF::NcFile::newFile);
        copyFromHyd(ncHyd, f);
        f.close();
#else
        copyFromHyd(ncHyd, ncFile);
#endif
        if(verbose) {
          std::cout << "Data copy complete." << std::endl;
        }
      }

#ifdef PARALLEL
      // Make sure all other processes wait
      mpiComm.Barrier();
      fMode = netCDF::NcFile::write;
#endif
    }
#ifdef PARALLEL
    netCDF::ParNcFile ncFile(mpiComm, mpiInfo, output_file, fMode);
#endif
    if(verbose) {
      std::cout << "Opened output file " << output_file << std::endl;
    }

    netCDF::NcDim timeDim, sigmaDim, cellDim;

    // Initialize the netCDF file (or at least fetch dimensions)
    timeDim = ncFile.addDim("time", times);
    sigmaDim = ncFile.getDim("siglay");
    if(sigmaDim.isNull())
      sigmaDim = ncFile.addDim("siglay", input_files.size());
    cellDim = ncFile.getDim("node");
    if(cellDim.isNull())
      cellDim = ncFile.addDim("node", nodes);
    netCDF::NcVar timeVar = ncFile.addVar("time", netCDF::NcType::nc_FLOAT,
        timeDim);

    // allVars will be a map of all the output variables, keyed by
    // offset in the history file
    std::map<size_t, netCDF::NcVar> allVars = assembleVars(ncFile, config);
    // This array tracks where variables without depth dependence are first
    // found. Sediment variables are only in
    // the last file so we may have to keep checking for them
    std::map<size_t, size_t> vars_found;

    Timer timer;

    for(size_t i = 0; i < input_files.size(); ++i) {
      try {
        if(verbose) {
          std::cout << "=== " << input_files[i] << " ===" << std::endl;
        }
        // when parallel, have rank 0 construct without a statevars
        // argument, then broadcast statevars to the others so they can
        // construct with that parameter known. That way all processes
        // don't have to do a bunch of I/O, just one
        unsigned long our_state_vars;
        if(rank == 0) {
          hf.set_file(input_files[i]);
          our_state_vars = hf.get_statevars();
        }
#ifdef PARALLEL
        mpiComm.Bcast(&our_state_vars, 1, MPI::UNSIGNED_LONG, 0);
        if(rank > 0) {
          hf.set_file(input_files[i], our_state_vars);
        }
#endif

        // MPI-parallelized loop
        for(size_t t = rank; t < hf.get_times(); t+=procs) {
          timer.start();
          time = hf.read_time(t);

          if(verbose) {
            std::cout << "file:" << std::setw(input_w) << i + 1;
            std::cout << "/" << input_files.size() << " time:" << std::setw(6) << time;
            std::cout << " (" << std::setw(t_w) << t + 1 << "/" << times;
            std::cout << ")..." << std::flush;
          }

          // Convert time from days to seconds in the cdf so it's
          // consistent with the native netcdf output
          timeVar.putVar({t}, time * 86400);
          unsigned long bytes_written = sizeof(time);

          // Read the data we want to output to the NetCDF file.
          for(auto& [data_index, v] : allVars) {
            bool has_time = true, has_sigma = true;
            // Only extract time-independent data in the first timestep
            if(v.getDim(0) != timeDim) {
              if(t > 0) {
                continue;
              }
              has_time = false;
            }
            // Only extract depth-independent data in the first layer file
            // where it's available
            std::vector<netCDF::NcDim> dims = v.getDims();
            if(std::find(dims.begin(), dims.end(), sigmaDim) == dims.end()) {
              if(! vars_found.contains(data_index)) { // C++20 feature
                if(data_index + 1 > our_state_vars) {
                  // Not available in this file
                  continue;
                }
                // It's available here, so mark it
                vars_found[data_index] = i;
              } else if(vars_found[data_index] != i) {
                // Already found in another layer; skip
                continue;
              }
              // Go ahead and extract
              has_sigma = false;
            }

            std::vector<float> data = hf.read_statevar(t, data_index);

            if(has_time && has_sigma) {
              // Write this variable at all the cells in this layer at this
              // time
              v.putVar({t, i, 0}, {1, 1, nodes}, data.data());
            } else if(has_time) {
              // No sigma info
              v.putVar({t, 0}, {1, nodes}, data.data());
            } else if(has_sigma) {
              // No time info
              v.putVar({i, 0}, {1, nodes}, data.data());
            } else {
              // No time or sigma info; one dimensional data
              v.putVar({0}, {nodes}, data.data());
            }
            bytes_written += sizeof(float) * data.size();
          }
          timer.stop();
          if(verbose) {
            // Determine the number of processes that actually did work
            const size_t p = std::min(procs, hf.get_times() - t);

            const int ms_per_it = std::floor(timer.elapsedMilliseconds()/p);
            std::cout << ms_per_it << " ms/it; ";
            const int kbps = std::floor(bytes_written * p / 1024
                / timer.elapsedSeconds());
            std::cout << kbps << " KiBps" << std::endl;
          }
        }
#ifdef PARALLEL
        // Let all the processes catch up before moving on to the next
        // file
        mpiComm.Barrier();
#endif
      } catch(std::ifstream::failure& e) {
        std::cerr << "Complete parsing for " << input_files[i]
            << " failed at time " << time << std::endl;
        std::cerr << e.what() << std::endl;
        abort(1);
      } catch(HistoryFileException& e) {
        std::cerr << "Error in file " << e.getHistoryFile()->get_path() << std::endl;
        std::cerr << e.what() << std::endl;
        abort(1);
      }
    }
#ifdef PARALLEL
    ncFile.close();
    MPI::Finalize();
  } catch(MPI::Exception& e) {
    std::cout << "MPI error: " << e.Get_error_string() << std::endl;
    mpiComm.Abort(-1);
#endif
  } catch(netCDF::exceptions::NcException& e) {
    std::cerr << e.what() << std::endl;
    abort(1);
  } catch(HistoryFileException& e) {
    std::cerr << "Error in file " << e.getHistoryFile()->get_path() << std::endl;
    std::cerr << e.what() << std::endl;
    abort(1);
  }
  return 0;
}
// vim: set shiftwidth=2 expandtab:
