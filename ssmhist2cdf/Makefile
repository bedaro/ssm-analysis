CXX := $(shell nc-config --cxx4)
#OPT := -O0 -g
OPT := -O3 -march=native -DNDEBUG
CXXFLAGS += $(OPT) -std=c++20 -Wall $(shell nc-config --cxx4flags)
NETCDF_LIBS = -Wl,-Bdynamic $(shell nc-config --cxx4libs)
LIBS := -Wl,-Bstatic -lboost_program_options -lyaml-cpp $(NETCDF_LIBS) -Wl,--as-needed
export NETCDF_LIBS
export LIBS
export CXXFLAGS

TARGETS := ssmhist2cdf
OBJS := ssmhist2cdf.o assembleVars.o copyFromHyd.o historyfile.o

ifdef PARALLEL
	CXX := mpicxx
	CXXFLAGS += -DPARALLEL

	OBJS += parNcFile.o
endif


all: $(TARGETS) test

ssmhist2cdf: $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LIBS)

test:
	make -C tests all

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

clean:
	/bin/rm -f $(TARGETS) *.o *~
	make -C tests clean

# Dependencies (hardcoded because they're quite simple)
ssmhist2cdf.o: ssmhist2cdf.cpp parNcFile.h assembleVars.h copyFromHyd.h historyfile.hpp timer.hpp
parNcFile.o: parNcFile.cpp parNcFile.h
assembleVars.o: assembleVars.cpp assembleVars.h
copyFromHyd.o: copyFromHyd.cpp copyFromHyd.h
historyfile.o: historyfile.cpp historyfile.hpp
