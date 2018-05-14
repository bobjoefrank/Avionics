#ifndef classifier_h
#define classifier_h

#include <stdio.h>
#include <iostream>
#include <map>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

cv::Mat CropOcrImage(std::vector<std::vector<cv::Point> >, int, cv::Mat, std::vector<cv::Vec4i> hierarchy, int letter_min_area);

cv::Mat ResizeOcrImage(cv::Mat, int);

cv::Mat RotateOcrImage45(cv::Mat);

std::string findColor(cv::Mat, int);

#endif
