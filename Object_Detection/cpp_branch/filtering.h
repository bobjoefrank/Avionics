#ifndef _padded_roi_h
#define _padded_roi_h

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

cv::Mat getPaddedROI(const cv::Mat &input, int top_left_x, int top_left_y, int width, int height, cv::Scalar paddingColor);

std::vector<cv::KeyPoint> GroupKeypoints(std::vector<cv::KeyPoint>, int);

std::vector<cv::KeyPoint> DeleteDupes(std::vector<cv::KeyPoint>);

#endif
