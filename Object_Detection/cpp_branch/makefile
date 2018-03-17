CXX = clang++
OBJS = OGdetect.o orientation.o classifier.o padded_roi.o
DEPS = orientation.h classifier.h padded_roi.h
EXE = OGdetect
CFLAGS = `python3-config --cflags` -O3
LDFLAGS = `python3-config --ldflags`

all: OGdetect

clean:
	rm -rf $(OBJS) $(EXE)

$(EXE): $(OBJS) $(pkg-config --libs opencv)
	$(CXX) $(LDFLAGS) $(FLAGS) $(OBJS) -o $(EXE) $$(pkg-config --libs opencv)

 ##### OBJECT FILES #####

OGdetect.o: OGdetect.cpp $(DEPS)
	$(CXX) -c OGdetect.cpp -o OGdetect.o -ansi -Wall $(CFLAGS)

padded_roi.o: padded_roi.cpp padded_roi.h
	$(CXX) -c padded_roi.cpp -o padded_roi.o -ansi -Wall $(CFLAGS)

orientation.o: orientation.cpp orientation.h
	$(CXX) -c orientation.cpp -o orientation.o -ansi -Wall $(CFLAGS)

classifier.o: classifier.cpp classifier.h
	$(CXX) -c classifier.cpp -o classifier.o -ansi -Wall $(CFLAGS)
