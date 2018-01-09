#ifndef orientation_h
#define orientation_h

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void drawAxis(Mat&, Point, Point, Scalar, const float);

double getAngle(const vector<Point> &, Mat&, int, int);

#endif
