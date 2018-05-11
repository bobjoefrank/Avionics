CXX = clang++
OBJS = OGdetect.o orientation.o classifier.o filtering.o
DEPS = orientation.h classifier.h filtering.h
EXE = OGdetect
CFLAGS = -std=c++14 `python3-config --cflags` -O3 -Iextern -I/usr/local/include
LDFLAGS = `python3-config --ldflags` $$(pkg-config --libs opencv)

all: OGdetect

clean:
	rm -rf $(OBJS) $(EXE)

$(EXE): $(OBJS)
	$(CXX) $(LDFLAGS) $(FLAGS) $(OBJS) -o $(EXE)

 ##### OBJECT FILES #####

OGdetect.o: OGdetect.cpp $(DEPS)
	$(CXX) -c OGdetect.cpp -o OGdetect.o $(CFLAGS)

filtering.o: filtering.cpp filtering.h
	$(CXX) -c filtering.cpp -o filtering.o $(CFLAGS)

orientation.o: orientation.cpp orientation.h
	$(CXX) -c orientation.cpp -o orientation.o $(CFLAGS)

classifier.o: classifier.cpp classifier.h
	$(CXX) -c classifier.cpp -o classifier.o $(CFLAGS)
