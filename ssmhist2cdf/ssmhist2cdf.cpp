#include <iostream>
#include <fstream>
#include <filesystem> // Requires C++17
#include <vector>
#include <netcdf>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_repeat.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/support_line_pos_iterator.hpp>
#include <boost/program_options.hpp>
#ifdef PARALLEL
  #include <mpi.h>
  #include "parNcFile.h"

  // Check for C++ bindings (OpenMPI, Intel/MPICH)
  #if (OMPI_BUILD_CXX_BINDINGS == 0) || MPICH_SKIP_MPICXX
    #error "OpenMPI C++ bindings are missing, cannot compile"
  #endif

  #define abort(val) MPI::COMM_WORLD.Abort(-1)
#else
  #define abort(val) return val
#endif

#define STATEVARS 42
#define STATEVARS_BOTTOM 94
#define FILE_VER 4 // The hardcoded integer present in the header
#define BYTES_PER_FLOAT 15 // The size of each float field. The first
                           // one on each line is one shorter which
                           // compensates for the newlines

#define USAGE "Usage: ssmhist2cdf [options] output-file input-file [input-file ...]"

// depth is at index 3
// DO is at index 4
// lDOC is at index 5
// Alg1 is at index 6
// Alg2 is at index 7
// NH4 is at index 8
// NO3 is at index 9
// PO4 is at index 10
// Temp is at index 12
// Salt is at index 13
// rDOC is at index 15
// lPOC is at index 16
// rPOC is at index 17
// DIC is at index 20
// TALK is at index 21
// pH is at index 22
// pCO2 is at index 23

static const std::array<int, 17> output_indices({
  3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 20, 21, 22, 23
});
static const std::array<std::string, 17> output_variables({
  "depth", "DOXG", "LDOC", "B1", "B2", "NH4", "NO3", "PO4", "temp",
  "salinity", "RDOC", "LPOC", "RPOC", "TDIC", "TALK", "pH", "pCO2"
});

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
    ("state-vars", po::value<unsigned int>()->default_value(STATEVARS),
        "Number of state variables per cell")
    ("state-vars-bottom",
        po::value<unsigned int>()->default_value(STATEVARS_BOTTOM),
        "Number of state variables per cell on the bottom")
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

  const std::vector<std::string> input_files =
      vm["input-file"].as< std::vector<std::string> >();
  const std::string output_file = vm["output-file"].as<std::string>();
  const bool last_is_bottom = vm.count("last-is-bottom");
  bool verbose = vm.count("verbose");
  const unsigned int statevars = vm["state-vars"].as<unsigned int>(),
      statevars_bottom = vm["state-vars-bottom"].as<unsigned int>();
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
    std::vector<netCDF::NcDim> allDims(3);
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

    allDims[0] = timeDim;
    allDims[1] = sigmaDim;
    allDims[2] = cellDim;
    std::vector<netCDF::NcVar> allVars(output_variables.size());
    for(size_t j = 0; j < output_variables.size(); ++j) {
      allVars[j] = ncFile.addVar(output_variables[j],
                                 netCDF::NcType::nc_FLOAT, allDims);
    }

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
          for(size_t j = 0; j < output_indices.size(); ++j) {
            size_t data_index = output_indices[j];
            // Skip over:
            // all previous time blocks (t * bytes_per_time)
            // this time block's header (header_length)
            // all the state variables preceding the current index
            //   (data_index * bytes_per_statevar)
            level.seekg(t * bytes_per_time + header_length +
                data_index * bytes_per_statevar);
            level.read(buffer, bytes_per_statevar);
            if(! level) {
              // FIXME more informative error
              throw "Error reading enough bytes for an output index";
            }
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

            netCDF::NcVar v = allVars[j];
            // Write this variable at all the cells in this layer at this
            // time
            v.putVar({t, i, 0}, {1, 1, nodes}, data.data());
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
      } catch(std::ifstream::failure&) {
        std::cerr << "Complete parsing for " << input_files[i]
            << " failed at time " << time << std::endl;
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
