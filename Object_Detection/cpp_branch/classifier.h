#ifndef classifier_h
#define classifier_h

#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

cv::Mat createOcrImage(std::vector<std::vector<cv::Point> >, int, cv::Mat);

std::string findColor(cv::Mat, int);

std::vector<cv::Point> getMaxContour(cv::Mat&, int, int);

#endif
