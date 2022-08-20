CXX := $(shell nc-config --cxx4)
#OPT := -O0 -g
OPT := -O3 -march=native -DNDEBUG
CXXFLAGS += $(OPT) -std=c++17 -Wall $(shell nc-config --cxx4flags)
LIBS := -Wl,-Bstatic -lboost_program_options -Wl,-Bdynamic $(shell nc-config --cxx4libs) -Wl,--as-needed

TARGETS := ssmhist2cdf
OBJS := ssmhist2cdf.o

ifdef PARALLEL
	CXX := mpicxx
	CXXFLAGS += -DPARALLEL

	OBJS += parNcFile.o
endif


all: $(TARGETS)

ssmhist2cdf: $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LIBS)

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

clean:
	/bin/rm -f $(TARGETS) *.o *~

# Dependencies (hardcoded because they're quite simple)
ssmhist2cpp.cpp: parNcFile.h
parNcFile.cpp: parNcFile.h
