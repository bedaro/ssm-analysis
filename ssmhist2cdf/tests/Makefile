TARGETS = copyFromHyd_test

all: $(TARGETS)

COPYFROMHYD_OBJS = copyFromHyd_test.o ../copyFromHyd.o
copyFromHyd_test: $(COPYFROMHYD_OBJS)
	$(CXX) $(COPYFROMHYD_OBJS) -o $@ $(NETCDF_LIBS)
	./$@

clean:
	/bin/rm -f $(TARGETS) *.o *~

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<
