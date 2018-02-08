FLAGS = -pedantic -Wall -Wextra -Weffc++
OBJS = OGdetect.o padded_roi.o orientation.o classifier.o
DEPS = padded_roi.h orientation.h classifier.h
EXE = OGdetect
CFLAGS = -I/Library/Frameworks/Python.framework/Versions/3.5/include/python3.5m \
			-I/Library/Frameworks/Python.framework/Versions/3.5/include/python3.5m \
			-fno-strict-aliasing -Wsign-compare -fno-common -dynamic -DNDEBUG -g -fwrapv \
			-O3 -Wall -Wstrict-prototypes -arch i386 -arch x86_64
LDFLAGS = -L/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/config-3.5m -lpython3.5m -ldl -framework CoreFoundation
all: OGdetect

clean:
	rm -rf $(OBJS) $(EXE)

$(EXE): $(OBJS) $(pkg-config --libs opencv)
	g++ $(LDFLAGS) $(FLAGS) $(OBJS) -o $(EXE) $$(pkg-config --libs opencv)

##### OBJECT FILES #####

OGdetect.o: OGdetect.cpp $(DEPS)
	g++ -c OGdetect.cpp -o OGdetect.o -ansi -Wall $(CFLAGS)

padded_roi.o: padded_roi.cpp padded_roi.h
	g++ -c padded_roi.cpp -o padded_roi.o -ansi -Wall $(CFLAGS)

orientation.o: orientation.cpp orientation.h
	g++ -c orientation.cpp -o orientation.o -ansi -Wall $(CFLAGS)

classifier.o: classifier.cpp classifier.h
	g++ -c classifier.cpp -o classifier.o -ansi -Wall $(CFLAGS)
