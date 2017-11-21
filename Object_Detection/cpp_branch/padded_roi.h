#ifndef _padded_roi
#define _padded_roi

#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

Mat getPaddedROI(const Mat &input, int top_left_x, int top_left_y, int width, int height, Scalar paddingColor);

#endif
