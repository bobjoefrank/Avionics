CXX = clang++
OBJS = OGdetect.o orientation.o classifier.o filtering.o
DEPS = orientation.h classifier.h filtering.h
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

filtering.o: filtering.cpp filtering.h
	$(CXX) -c filtering.cpp -o filtering.o -ansi -Wall $(CFLAGS)

orientation.o: orientation.cpp orientation.h
	$(CXX) -c orientation.cpp -o orientation.o -ansi -Wall $(CFLAGS)

classifier.o: classifier.cpp classifier.h
	$(CXX) -c classifier.cpp -o classifier.o -ansi -Wall $(CFLAGS)
