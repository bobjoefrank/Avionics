FLAGS = -pedantic -Wall -Wextra -Weffc++
OBJS = roi_keypoint.o padded_roi.o orientation.o shape_classifier.o
DEPS = padded_roi.h orientation.h shape_classifier.h
EXE = roi_keypoint


all: roi_keypoint

clean:
	rm -rf $(OBJS) $(EXE)

roi_keypoint: $(OBJS) $(pkg-config --libs opencv)
	g++ $(FLAGS) $(OBJS) -o roi_keypoint $$(pkg-config --libs opencv)

##### OBJECT FILES #####

roi_keypoint.o: roi_keypoint.cpp $(DEPS)
	g++ -c roi_keypoint.cpp -o roi_keypoint.o -ansi -Wall

padded_roi.o: padded_roi.cpp padded_roi.h
	g++ -c padded_roi.cpp -o padded_roi.o -ansi -Wall

orientation.o: orientation.cpp orientation.h
	g++ -c orientation.cpp -o orientation.o -ansi -Wall

shape_classifier.o: shape_classifier.cpp shape_classifier.h
	g++ -c shape_classifier.cpp -o shape_classifier.o -ansi -Wall
