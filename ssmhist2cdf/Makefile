CXX := $(shell nc-config --cxx4)
CXXFLAGS += -O3 -march=native -std=c++11 -Wall $(shell nc-config --cxx4flags)
LIBS := $(shell nc-config --cxx4libs)

TARGETS := ssmhist2cdf

all: $(TARGETS)

ssmhist2cdf: ssmhist2cdf.cpp
	$(CXX) $(CXXFLAGS) $@.cpp -o $@ $(LIBS)

clean:
	/bin/rm -f $(TARGETS) *~
