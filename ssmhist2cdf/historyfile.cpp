#include "historyfile.hpp"
#include <iostream>
#include <filesystem> // Requires C++17
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_repeat.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/support_line_pos_iterator.hpp>

template <typename It>
bool HistoryFile::parse_statevar(It& pos, It last, std::vector<float>& data) {
  namespace qi = boost::spirit::qi;
  namespace phoenix = boost::phoenix;
  using phoenix::ref;
  using boost::spirit::repeat;
  using qi::_1;
  using qi::double_;
  using phoenix::push_back;
  using boost::spirit::ascii::space;

  return qi::phrase_parse(pos, last, (
    repeat(ref(nodes))[
      double_[push_back(phoenix::ref(data), _1)]
    ]), space);
}

size_t HistoryFile::read_header(float& hdr_time, size_t& hdr_node_ct) {
  int ver;
  std::string line;
  std::getline(stream, line);
  std::istringstream s(line);
  s >> hdr_time >> ver >> hdr_node_ct;
  if(ver != HISTORY_FILE_VER) {
    throw "Parse failed: version mismatch (" + std::to_string(ver) + ") in header";
  }
  // Include the newline in the total header size
  return line.length() + 1;
}
HistoryFile::HistoryFile() {
  stream.exceptions(std::ifstream::failbit|std::ifstream::badbit);
}

HistoryFile::HistoryFile(const std::string file_path) : HistoryFile() {
  set_file(file_path);
}

HistoryFile::HistoryFile(const std::string file_path,
    const size_t state_vars) : HistoryFile() {
  set_file(file_path, state_vars);
}

void HistoryFile::set_file(std::string file_path) {
  path = file_path;
  if(stream.is_open()) {
    stream.close();
  }
  stream.open(file_path);
  statevars = 0;
  getProperties();
}

void HistoryFile::set_file(std::string file_path, const size_t state_vars) {
  path = file_path;
  if(stream.is_open()) {
    stream.close();
  }
  stream.open(file_path);
  statevars = state_vars;
  getProperties();
}

size_t HistoryFile::get_statevars() { return statevars; }
size_t HistoryFile::get_nodes() { return nodes; }
size_t HistoryFile::get_times() { return times; }

void HistoryFile::getProperties() {
  using std::filesystem::file_size;

  stream.seekg(0);
  float first_time;
  header_length = read_header(first_time, nodes);
  if(statevars == 0) {
    // Read the entire first time block to figure out how big it is and
    // calculate the number of state variables
    // We know we've reached the next time header because the second
    // character on the line will be a space and the total line length
    // will equal the original header length
    size_t bpt = 0;
    while(true) {
      std::string line;
      std::getline(stream, line);
      if((line[1] == ' ') && (line.length() + 1 == header_length)) {
        break;
      }
      bpt += line.length() + 1;
    }
    // FIXME check that this divides evenly
    statevars = bpt / HIST_BYTES_PER_FLOAT / nodes;
#ifndef NDEBUG
    std::cout << "statevars is " << std::to_string(statevars) << std::endl;
#endif
    bytes_per_time = bpt + header_length;
  } else {
    // Based on the number of nodes in the model, the state variable
    // count and the size of each field, we can compute the number of
    // bytes each time block takes up.
    bytes_per_time = header_length + HIST_BYTES_PER_FLOAT * nodes *
      statevars;
  }
  // Check the input file size to infer how many times there are
  size_t sz = file_size(path);
  if(sz % bytes_per_time != 0) {
    throw "File is malformed, size is not a multiple of time blocks";
  }
  times = sz / bytes_per_time;
}

float HistoryFile::read_time(size_t t) {
  // Seek to the beginning of the time block to read the header
  stream.seekg(t * bytes_per_time);
  float time;
  read_header(time, nodes);
  return time;
}

std::vector<float> HistoryFile::read_statevar(size_t t, size_t i) {
  using boost::spirit::line_pos_iterator;
  const size_t bytes_per_statevar = HIST_BYTES_PER_FLOAT * nodes;
  if(read_buffer == nullptr) {
    // Initialize a read buffer to store (# nodes) floats for
    // a single state variable
    read_buffer = new char[bytes_per_statevar];
  }
  // Skip over:
  // all previous time blocks (t * bytes_per_time)
  // this time block's header (header_length)
  // all the state variables preceding the current index
  //   (i * bytes_per_statevar)
  stream.seekg(t * bytes_per_time + header_length +
      i * bytes_per_statevar);
  stream.read(read_buffer, bytes_per_statevar);
  // Connect Spirit-compatible iterators to the buffer for
  // parsing. See
  // https://www.boost.org/doc/libs/1_71_0/libs/spirit/doc/html/spirit/support/line_pos_iterator.html
  line_pos_iterator<char*> begin(read_buffer),
    end(read_buffer + bytes_per_statevar);
  std::vector<float> data;
  if(! parse_statevar(begin, end, data)) {
    throw "Parse failed!";
  }
  if(data.size() != nodes) {
    throw "Parse did not complete (" + std::to_string(data.size()) +
      " < " + std::to_string(nodes) + " nodes)";
  }
  return data;
}
// vim: set shiftwidth=2 expandtab:
