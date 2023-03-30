#include "../historyfile.hpp"
#define BOOST_TEST_MODULE history_file
#include <boost/test/included/unit_test.hpp>

// The sample file is set up for 20 nodes, and has 60 data values
// per output time. That means there are 60/20 == 3 state variables.
// Four output times are given.
#define SAMPLE_FILENAME "sample_history_file.out"

BOOST_AUTO_TEST_CASE( builtin_ifstream ) {
  namespace tt = boost::test_tools;
  HistoryFile hf1 = HistoryFile(SAMPLE_FILENAME);
  BOOST_CHECK_EQUAL(hf1.get_statevars(), 3);
  BOOST_CHECK_EQUAL(hf1.get_nodes(), 20);
  BOOST_CHECK_EQUAL(hf1.get_times(), 4);

  // Check times
  BOOST_CHECK_EQUAL(hf1.read_time(0), 0.25);
  BOOST_CHECK_EQUAL(hf1.read_time(1), 0.5);
  BOOST_CHECK_EQUAL(hf1.read_time(2), 0.75);
  BOOST_CHECK_EQUAL(hf1.read_time(3), 1);

  // Check reading state variables. Second timestep,
  // second statevar
  std::vector<float> data = hf1.read_statevar(1, 1);
  BOOST_CHECK_EQUAL(data.size(), 20);
  BOOST_TEST(data[1] == 140.436, tt::tolerance(0.0001));
  BOOST_TEST(data[5] == 102.757, tt::tolerance(0.0001));

  // One more statevar. Last timestep, last statevar
  data = hf1.read_statevar(3, 2);
  BOOST_CHECK_EQUAL(data.size(), 20);
  BOOST_CHECK_EQUAL(data[2], 480);
  BOOST_TEST(data[8] == 440.358, tt::tolerance(0.0001));
}

BOOST_AUTO_TEST_CASE( force_statevars ) {
  HistoryFile hf1 = HistoryFile(SAMPLE_FILENAME, 3);
  BOOST_CHECK_EQUAL(hf1.get_statevars(), 3);
  BOOST_CHECK_EQUAL(hf1.get_nodes(), 20);
  BOOST_CHECK_EQUAL(hf1.get_times(), 4);
}

// TODO test case with a second sample file for object reuse with set_file()

// vim: set shiftwidth=2 expandtab:
