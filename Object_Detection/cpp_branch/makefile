FLAGS = -pedantic -Wall -Wextra -Weffc++
OBJS = roi_keypoint.o padded_roi.o orientation.o shape_classifier.o
DEPS = padded_roi.h orientation.h shape_classifier.h
EXE = roi_keypoint
CFLAGS = -I/Library/Frameworks/Python.framework/Versions/3.5/include/python3.5m \
			-I/Library/Frameworks/Python.framework/Versions/3.5/include/python3.5m \
			-fno-strict-aliasing -Wsign-compare -fno-common -dynamic -DNDEBUG -g -fwrapv \
			-O3 -Wall -Wstrict-prototypes -arch i386 -arch x86_64 -g
LDFLAGS = -L/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/config-3.5m -lpython3.5m -ldl -framework CoreFoundation
all: roi_keypoint

clean:
	rm -rf $(OBJS) $(EXE)

roi_keypoint: $(OBJS) $(pkg-config --libs opencv)
	g++ $(LDFLAGS) $(FLAGS) $(OBJS) -o roi_keypoint $$(pkg-config --libs opencv)

##### OBJECT FILES #####

roi_keypoint.o: roi_keypoint.cpp $(DEPS)
	g++ -c roi_keypoint.cpp -o roi_keypoint.o -ansi -Wall $(CFLAGS)

padded_roi.o: padded_roi.cpp padded_roi.h
	g++ -c padded_roi.cpp -o padded_roi.o -ansi -Wall $(CFLAGS)

orientation.o: orientation.cpp orientation.h
	g++ -c orientation.cpp -o orientation.o -ansi -Wall $(CFLAGS)

shape_classifier.o: shape_classifier.cpp shape_classifier.h
	g++ -c shape_classifier.cpp -o shape_classifier.o -ansi -Wall $(CFLAGS)
