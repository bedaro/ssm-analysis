#include <iostream>
#include <fstream>
#include <filesystem> // Requires C++17
#include <vector>
#include <map>
#include <algorithm>
#include <netcdf>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_repeat.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/support_line_pos_iterator.hpp>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include "assembleVars.h"
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

#define FILE_VER 4 // The hardcoded integer present in the header
#define BYTES_PER_FLOAT 15 // The size of each float field. The first
                           // one on each line is one shorter which
                           // compensates for the newlines

#define USAGE "Usage: ssmhist2cdf [options] output-file input-file [input-file ...]"

// Default configuration
static const std::string default_variables =
  "state_vars: 42\n"
  "state_vars_bottom: 94\n"
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

namespace spirit = boost::spirit;
namespace po = boost::program_options;
namespace fs = std::filesystem;

/*
 * An approximately EBNF form of the file structure:
 * <file> ::= {<block>}
 * <block> ::= <time> <ver> <cell-count>\n
 *             cell-count*state_vars*{<double>{ |\n}}
 * <time> ::= <float>
 * <ver> ::= <int>
 * <cell-count> ::= <int>
 * Plainly, each block of the file consists of a header that contains
 * an integer cell count. What follows in the block is a repetition
 * of (cell count times state_vars) doubles, each followed by a
 * newline or one or more spaces.
 */

/*
 * Parse one state variable from the given iterators using Spirit.
 */
template <typename It>
bool parse_statevar(It& pos, It last, size_t nodes, std::vector<float>& data) {
  namespace qi = boost::spirit::qi;
  namespace phoenix = boost::phoenix;
  using phoenix::ref;
  using spirit::repeat;
  using qi::_1;
  using qi::double_;
  using phoenix::push_back;
  using spirit::ascii::space;

  return qi::phrase_parse(pos, last, (
    repeat(ref(nodes))[
      double_[push_back(phoenix::ref(data), _1)]
    ]), space);
}

/*
 * Parse the "header" lines in the file, validate the version, and
 * set the time/node count. Returns the number of bytes read in the
 * header
 */
size_t read_header(std::ifstream& f, float& time, size_t& nodes) {
  int ver;
  std::string line;
  std::getline(f, line);
  std::istringstream s(line);
  s >> time >> ver >> nodes;
  if(ver != FILE_VER) {
    throw "Parse failed: version mismatch (" + std::to_string(ver) + ") in header";
  }
  // Include the newline in the total header size
  return line.length() + 1;
}

int main(int argc, char *argv[]) {
  int rank = -1;
  float time = 0;
#ifndef PARALLEL
  rank = 0;
  const int procs = 1;
#else
  MPI::Init(argc, argv);
  MPI::Comm& mpiComm = MPI::COMM_WORLD;
  rank = mpiComm.Get_rank();
  const int procs = mpiComm.Get_size();
#endif

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("last-is-bottom",
        "treat the last file as the bottom layer with sediment state "
        "variables")
    ("config,c", po::value<std::string>(),
        "Specify a config file with the output variables")
    ("verbose,v", "Verbose output")
    ("output-file", po::value<std::string>()->required(),
        "Output netcdf file name")
    ("input-file", po::value< std::vector<std::string> >()->required(),
        "Input SSM history file(s) in layer order")
  ;
  po::positional_options_description p;
  p.add("output-file", 1).add("input-file", -1);

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).
              options(desc).positional(p).run(), vm);

    // Fix from https://stackoverflow.com/a/5517755
    if(vm.count("help") && (rank == 0)) {
      std::cout << USAGE << std::endl << desc << std::endl;
      abort(0);
    }

    po::notify(vm);
  } catch (boost::program_options::error& e) {
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
  const bool last_is_bottom = vm.count("last-is-bottom");
  bool verbose = vm.count("verbose");
  const unsigned int statevars = config["state_vars"].as<unsigned int>(),
      statevars_bottom = config["state_vars_bottom"].as<unsigned int>();
  // Field width of input file index
  int input_w = std::ceil(std::log10(input_files.size() + 1));

  try {
#ifndef PARALLEL
    netCDF::NcFile ncFile(output_file, netCDF::NcFile::newFile);
#else
    verbose &= (rank == 0);
    if(verbose) {
      std::cout << "Running in parallel, " << procs << " processes" << std::endl;
    }

    mpiComm.Set_errhandler(MPI::ERRORS_THROW_EXCEPTIONS);
    MPI::Info mpiInfo = MPI::INFO_NULL;

    netCDF::ParNcFile ncFile(mpiComm, mpiInfo, output_file, netCDF::NcFile::newFile);
#endif

    unsigned long nodes, times, header_length;
    netCDF::NcDim timeDim, sigmaDim, cellDim;
    std::ifstream level;
    level.exceptions(std::ifstream::failbit|std::ifstream::badbit);

    // Only do this once to ensure processes can't be out of sync due
    // to I/O based race conditions
    if(rank == 0) {
      // Read the first header line of the first file to get node count
      level.open(input_files[0]);
      header_length = read_header(level, time, nodes);
      level.close();

      // Based on the number of nodes in the model, the state variable
      // count and the size of each field, we can compute the number of
      // bytes each time block takes up.
      size_t bytes_per_time = header_length + BYTES_PER_FLOAT * nodes *
        ((input_files.size() == 1) && last_is_bottom? statevars_bottom : statevars);
      // Check the input file size to infer how many times there are
      times = fs::file_size(input_files[0]) / bytes_per_time;
    }
#ifdef PARALLEL
    // Distribute the history file properties to the other processes
    mpiComm.Bcast(&nodes, 1, MPI::UNSIGNED_LONG, 0);
    mpiComm.Bcast(&times, 1, MPI::UNSIGNED_LONG, 0);
    mpiComm.Bcast(&header_length, 1, MPI::UNSIGNED_LONG, 0);
#endif

    // Initialize the netCDF file
    timeDim = ncFile.addDim("time", times),
    sigmaDim = ncFile.addDim("siglay", input_files.size()),
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

    for(size_t i = 0; i < input_files.size(); ++i) {
      try {
        // The number of state variables
        unsigned int our_state_vars = ((i + 1 == input_files.size()) && last_is_bottom?
                statevars_bottom : statevars);
        // The byte size per time block, recomputed from earlier to allow
        // for the possibility that the number of state variables in this
        // file is different
        size_t bytes_per_time = header_length + BYTES_PER_FLOAT * nodes * our_state_vars;

        level.open(input_files[i]);
        if(verbose) {
          std::cout << "=== " << input_files[i] << " ===" << std::endl;
        }

        // MPI-parallelized loop
        for(size_t t = rank; t < times; t+=procs) {
          // Seek to the beginning of the time block to read the header
          level.seekg(t * bytes_per_time);
          read_header(level, time, nodes);

          if(verbose) {
            std::cout << "(";
            std::cout.width(input_w);
            std::cout << i + 1;
            std::cout.width(0);
            std::cout << "/" << input_files.size() << ") TIME " <<
              time << "... " << std::flush;
          }

          // Initialize a read buffer to store (# nodes) floats for
          // a single state variable
          const size_t bytes_per_statevar = BYTES_PER_FLOAT * nodes;
          char *buffer = new char[bytes_per_statevar];

          // Convert time from days to seconds in the cdf so it's
          // consistent with the native netcdf output
          timeVar.putVar({t}, time * 86400);

          // Read only the data we want to output to the NetCDF file.
          // We can do this by seeking within the file to locations where
          // we know this data can be found, rather than parsing the
          // entire time block.
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

            // Skip over:
            // all previous time blocks (t * bytes_per_time)
            // this time block's header (header_length)
            // all the state variables preceding the current index
            //   (data_index * bytes_per_statevar)
            level.seekg(t * bytes_per_time + header_length +
                data_index * bytes_per_statevar);
            level.read(buffer, bytes_per_statevar);
            // Connect Spirit-compatible iterators to the buffer for
            // parsing. See
            // https://www.boost.org/doc/libs/1_71_0/libs/spirit/doc/html/spirit/support/line_pos_iterator.html
            spirit::line_pos_iterator<char*> begin(buffer), end(buffer + bytes_per_statevar);
            std::vector<float> data;
            if(! parse_statevar(begin, end, nodes, data)) {
              throw "Parse failed!";
            }
            if(data.size() != nodes) {
              throw "Parse did not complete (" + std::to_string(data.size()) + " nodes)";
            }

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
          }
          if(verbose) {
            std::cout << " complete." << std::endl;
          }
        }
#ifdef PARALLEL
        // Let all the processes catch up before moving on to the next
        // file
        mpiComm.Barrier();
#endif
        level.close();
      } catch(std::ifstream::failure& e) {
        std::cerr << "Complete parsing for " << input_files[i]
            << " failed at time " << time << std::endl;
        std::cerr << e.what() << std::endl;
        abort(1);
      } catch(char const *msg) {
        std::cerr << msg << std::endl;
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
  } catch(char const *msg) {
    std::cerr << msg << std::endl;
    abort(1);
  }
  return 0;
}
// vim: set shiftwidth=2 expandtab:
