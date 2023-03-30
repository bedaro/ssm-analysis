TARGETS = copyFromHyd_test historyfile_test

all: $(TARGETS)

COPYFROMHYD_OBJS = copyFromHyd_test.o ../copyFromHyd.o
copyFromHyd_test: $(COPYFROMHYD_OBJS)
	$(CXX) $(COPYFROMHYD_OBJS) -o $@ $(NETCDF_LIBS)
	./$@

HISTORYFILE_OBJS = historyfile_test.o ../historyfile.o
historyfile_test: $(HISTORYFILE_OBJS)
	$(CXX) $(HISTORYFILE_OBJS) -o $@
	./$@

clean:
	/bin/rm -f $(TARGETS) *.o *~

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

copyFromHyd_test.o: ../copyFromHyd.h
historyfile_test.o: ../historyfile.hpp
