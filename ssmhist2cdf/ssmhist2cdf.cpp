#include <iostream>
#include <fstream>
#include <netcdf>
#include <vector>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_repeat.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/support_istream_iterator.hpp>
#include <boost/program_options.hpp>

#define STATEVARS 42
#define STATEVARS_BOTTOM 94
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

template <typename It>
bool parse_block(It& pos, It last, float& time, const unsigned int& state_vars, size_t& cells, std::vector<float>& data) {
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
  namespace qi = boost::spirit::qi;
  namespace phoenix = boost::phoenix;
  using phoenix::ref;
  using spirit::repeat;
  using qi::_1;
  using qi::float_;
  using qi::int_;
  using qi::double_;
  using phoenix::push_back;
  using spirit::ascii::space;

  return qi::phrase_parse(pos, last, (
        float_[ref(time) = _1] >> int_(4) >> int_[ref(cells) = _1] >>
        repeat(ref(cells))[
          repeat(ref(state_vars))[
            double_[push_back(phoenix::ref(data), _1)]
          ]
        ]), space);
}

int main(int argc, char *argv[]) {

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
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cerr << USAGE << std::endl
              << "Try ssmhist2cdf --help" << std::endl;
    return 1;
  }

  if(vm.count("help")) {
    std::cout << USAGE << std::endl << desc << std::endl;
    return 0;
  }

  const std::vector<std::string> input_files =
      vm["input-file"].as< std::vector<std::string> >();
  const std::string output_file = vm["output-file"].as<std::string>();
  const bool last_is_bottom = vm.count("last-is-bottom"),
             verbose = vm.count("verbose");
  const unsigned int statevars = vm["state-vars"].as<unsigned int>(),
      statevars_bottom = vm["state-vars-bottom"].as<unsigned int>();
  // Field width of input file index
  int input_w = std::ceil(std::log10(input_files.size() + 1));

  try {
    // Initialize the netCDF file
    netCDF::NcFile ncFile(output_file, netCDF::NcFile::newFile);
    netCDF::NcDim timeDim = ncFile.addDim("time"),
                  sigmaDim = ncFile.addDim("siglay", input_files.size()),
                  cellDim; // initialized later once we know how large
    std::vector<netCDF::NcDim> allDims(3);
    std::vector<netCDF::NcVar> allVars(output_variables.size());
    netCDF::NcVar timeVar = ncFile.addVar("time", netCDF::NcType::nc_FLOAT,
        timeDim);
    allDims[0] = timeDim;
    allDims[1] = sigmaDim;
    float time;
    for(size_t i = 0; i < input_files.size(); ++i) {
      try {
        std::ifstream level;
        level.exceptions(std::ifstream::failbit|std::ifstream::badbit);
        level.open(input_files[i]);
        if(verbose) {
          std::cout << "=== " << input_files[i] << " ===" << std::endl;
        }
        // Set up our stream iterators
        // http://boost-spirit.com/home/2010/01/05/stream-based-parsing-made-easy/
        level.unsetf(std::ios::skipws);
        spirit::istream_iterator pos(level);
        spirit::istream_iterator last;

        // The number of state variables
        unsigned int our_state_vars = ((i + 1 == input_files.size()) && last_is_bottom?
                statevars_bottom : statevars);
        time = 0; // initialize to 0 to accommodate an error check
        size_t cells; // the number of cells (read from the block)
        size_t t = 0; // the time index for the current file
        while(pos != last) {
          float prev_time = time;
          std::vector<float> data;
          if(! parse_block(pos, last, time, our_state_vars, cells, data)) {
            throw "Parse failed!";
          }
          // The parse has retrieved time, cells, and data. pos has been
          // moved to the end of the block.

          if(verbose) {
            std::cout << "(";
            std::cout.width(input_w);
            std::cout << i + 1;
            std::cout.width(0);
            std::cout << "/" << input_files.size() <<
              ") TIME " << time << "... " << std::flush;
          }
          if(prev_time > time) {
            if(verbose) {
              std::cout << std::endl;
            }
            // Something has gone horribly wrong in parsing, abort
            throw "Time is out of order, there must be a parse failure!";
          }
          // Convert time from days to seconds in the cdf so it's
          // consistent with the native netcdf output
          timeVar.putVar({t}, time * 86400);
          if((i == 0) && (t == 0)) {
            // For the first pass of the first file, we need to set
            // up some more CDF variables now that we know the number of
            // cells
            cellDim = ncFile.addDim("node", cells);
            allDims[2] = cellDim;
            for(size_t j = 0; j < output_variables.size(); ++j) {
              allVars[j] = ncFile.addVar(output_variables[j],
                  netCDF::NcType::nc_FLOAT, allDims);
            }
          }

          // Save the data to the CDF
          // Loop over all the variables to write
          float *data_raw = data.data();
          for(size_t j = 0; j < output_indices.size(); ++j) {
            netCDF::NcVar v = allVars[j];
            size_t data_index = output_indices[j];
            // Write this variable at all the cells in this layer at this
            // time
            v.putVar({t, i, 0}, {1, 1, cells}, data_raw + data_index * cells);
          }
          if(verbose) {
            std::cout << " complete." << std::endl;
          }
          ++t;
        }
      } catch(std::ifstream::failure&) {
        std::cerr << "Complete parsing for " << input_files[i]
            << " failed at time " << time << std::endl;
        return 1;
      } catch(char const *msg) {
        std::cerr << msg << std::endl;
        return 1;
      }
    }
  } catch(netCDF::exceptions::NcException& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
