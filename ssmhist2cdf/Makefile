CXX := c++
CXXFLAGS += -O3 -march=native -std=c++11 -Wall $(shell ncxx4-config --cflags)
LIBS := $(shell ncxx4-config --libs)

TARGETS := ssmhist2cdf

all: $(TARGETS)

ssmhist2cdf: ssmhist2cdf.cpp
	$(CXX) $(CXXFLAGS) $@.cpp -o $@ $(LIBS)

clean:
	/bin/rm -f $(TARGETS) *~
