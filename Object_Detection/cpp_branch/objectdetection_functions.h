#ifndef OBJECTDETECTION_FUNCTIONS_H
#define OBJECTDETECTION_FUNCTIONS_H

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

std::vector<cv::KeyPoint> get_SURF_detected_keypoints(const Mat img, int minHessian);
std::vector<cv::KeyPoint> group_keypoints(std::vector<cv::KeyPoint> keypoints, int keypoint_group_distance);
std::vector<cv::KeyPoint> delete_duplicates(std::vector<cv::KeyPoint> keypoints);
Mat getPaddedROI(const Mat &input, int top_left_x, int top_left_y, int width, int height, Scalar paddingColor);
void getKmeansImage(Mat &src, Mat &dst, Mat &centers, int k);
Mat getShapeAsBinary(const Mat kmeans_img, int roi_width);

#endif
