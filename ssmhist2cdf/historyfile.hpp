#ifndef HISTORYFILE_H
#define HISTORYFILE_H
#include <fstream>
#include <vector>

// The hardcoded integer present in the header
#define HISTORY_FILE_VER 4 

// The size of each float field. The first one on each line is one shorter
// which compensates for the newlines
#define HIST_BYTES_PER_FLOAT 15

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

class HistoryFile {
  public:
    HistoryFile() noexcept;
    HistoryFile(std::string file_path);
    HistoryFile(std::string file_path, size_t state_vars);
    float read_time(size_t t);
    std::vector<float> read_statevar(size_t t, size_t i);

    void set_file(std::string file_path);
    void set_file(std::string file_path, size_t state_vars);
    std::string get_path() const noexcept;
    size_t get_statevars() const noexcept;
    size_t get_nodes() const noexcept;
    size_t get_times() const noexcept;

  private:
    std::string path;
    std::ifstream stream;
    size_t statevars = 0, nodes, times, header_length,
           bytes_per_time, bytes_per_float;
    char *read_buffer = nullptr;

    /*
     * Parse one state variable from the given iterators using Spirit.
     */
    template <typename It> bool parse_statevar(It& pos, It last,
        std::vector<float>& data) const noexcept;

    /*
     * Parse the "header" lines in the file, validate the version, and
     * set the time/node count. Returns the number of bytes read in the
     * header
     */
    size_t read_header(float& hdr_time, size_t& hdr_node_ct);

    void getProperties();

};

class HistoryFileException : public std::exception {
  protected:
    const HistoryFile* history_file;
    std::string error_message;
    int error_offset;
  public:
    explicit HistoryFileException(const HistoryFile* hf, const std::string& msg):
      history_file(hf),
      error_message(msg),
      error_offset(-1)
      {}

    explicit HistoryFileException(const HistoryFile* hf, const std::string& msg,
        int err_off):
      history_file(hf),
      error_message(msg),
      error_offset(err_off)
      {}

    virtual const char *what() const noexcept { return error_message.c_str(); }
    virtual const HistoryFile* getHistoryFile() const noexcept { return history_file; }
    virtual int getErrorOffset() const noexcept { return error_offset; }
};

#endif
// vim: set shiftwidth=2 expandtab:
