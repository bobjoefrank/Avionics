#ifndef orientation_h
#define orientation_h

#include <iostream>
#include <opencv2/opencv.hpp>

void drawAxis(cv::Mat&, cv::Point, cv::Point, cv::Scalar, const float);

double getAngle(const std::vector<cv::Point> &, cv::Mat&, int, int);

#endif
