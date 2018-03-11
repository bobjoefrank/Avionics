FLAGS = -pedantic -Wall -Wextra -Weffc++ -g -std=c++14
OBJS = OGdetect.o padded_roi.o orientation.o classifier.o keras_model.o
DEPS = padded_roi.h orientation.h classifier.h keras_model.h
EXE = OGdetect
CFLAGS = $(FLAGS) `python3-config --cflags`
LDFLAGS = `python3-config --ldflags`

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

keras_model.o: keras_model.cpp keras_model.h
	g++ -c keras_model.cpp -o keras_model.o -ansi -Wall $(CFLAGS)
