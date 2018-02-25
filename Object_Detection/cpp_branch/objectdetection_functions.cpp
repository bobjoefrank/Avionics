#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

#include "objectdetection_functions.h"

using namespace cv;

std::vector<cv::KeyPoint> get_SURF_detected_keypoints(const Mat img, int minHessian) {

    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );
    std::vector<cv::KeyPoint> keypoints;
    detector->detect( img, keypoints);

    return keypoints;

}


std::vector<cv::KeyPoint> group_keypoints(std::vector<cv::KeyPoint> keypoints, int keypoint_group_distance) {

    std::vector<cv::KeyPoint> grouped_keypoints = keypoints;

    int index_a = 0;
    for(std::vector<cv::KeyPoint>::const_iterator point_a = grouped_keypoints.begin(); point_a != grouped_keypoints.end(); point_a++){
        int index_b = 0;
        for(std::vector<cv::KeyPoint>::const_iterator point_b = grouped_keypoints.begin(); point_b != grouped_keypoints.end(); point_b++){
            if (abs(point_a->pt.x - point_b->pt.x) <= keypoint_group_distance && abs(point_a->pt.y - point_b->pt.y) <= keypoint_group_distance){
                if (point_b->response > point_a->response){
                    grouped_keypoints[index_a] = grouped_keypoints[index_b];
                }
            }
            index_b += 1;
        }
        index_a += 1;
    }

    return grouped_keypoints;

    // alternate cluster keypoint without stranded keypoints
    /*std::vector<cv::KeyPoint> grouped_keypoints;
    int index_a = 0;
    int max_index = 0;
    for(std::vector<cv::KeyPoint>::const_iterator point_a = keypoints.begin(); point_a != keypoints.end(); point_a++){
        int index_b = 0;
        float max_response = 0;
        for(std::vector<cv::KeyPoint>::const_iterator point_b = keypoints.begin(); point_b != keypoints.end(); point_b++){
            if (abs(point_a->pt.x - point_b->pt.x) <= keypoint_group_distance && abs(point_a->pt.y - point_b->pt.y) <= keypoint_group_distance){
                if (point_b->response > max_response){
                    max_response = point_b->response;
                    max_index = index_b;
                }
            }
            index_b += 1;
        }
        grouped_keypoints.push_back(keypoints[max_index]);
        index_a += 1;
    }*/
}


std::vector<cv::KeyPoint> delete_duplicates(std::vector<cv::KeyPoint> keypoints){

    std::vector<cv::KeyPoint> removed_duplicates = keypoints;
    unsigned long j = 0;
    while(j < removed_duplicates.size()-1) {
        for(size_t i= j+1 ; i < removed_duplicates.size();){
            if( removed_duplicates[j].pt.x == removed_duplicates[i].pt.x ){
                removed_duplicates.erase(removed_duplicates.begin()+i);
            } else {
                ++i;
            }
        }
        ++j;
    }

    return removed_duplicates;
}


Mat getPaddedROI(const Mat &input, int top_left_x, int top_left_y, int width, int height, Scalar paddingColor){
    int bottom_right_x = top_left_x + width;
    int bottom_right_y = top_left_y + height;

    Mat output;
    if (top_left_x < 0 || top_left_y < 0 || bottom_right_x > input.cols || bottom_right_y > input.rows) {
        // border padding will be required
        int border_left = 0, border_right = 0, border_top = 0, border_bottom = 0;

        if (top_left_x < 0) {
            width = width + top_left_x;
            border_left = -1 * top_left_x;
            top_left_x = 0;
        }
        if (top_left_y < 0) {
            height = height + top_left_y;
            border_top = -1 * top_left_y;
            top_left_y = 0;
        }
        if (bottom_right_x > input.cols) {
            width = width - (bottom_right_x - input.cols);
            border_right = bottom_right_x - input.cols;
        }
        if (bottom_right_y > input.rows) {
            height = height - (bottom_right_y - input.rows);
            border_bottom = bottom_right_y - input.rows;
        }

        Rect R(top_left_x, top_left_y, width, height);
        copyMakeBorder(input(R), output, border_top, border_bottom, border_left, border_right, BORDER_CONSTANT, paddingColor);
    }
    else {
        // no border padding required
        Rect R(top_left_x, top_left_y, width, height);
        output = input(R);
    }
    return output;
}


void getKmeansImage(Mat &src, Mat &dst, Mat &centers, int k) {
    int attempts = 5;
    cv::Mat labels;

    cv::Mat samples(src.rows * src.cols, 3, CV_32FC2);
    for( int y = 0; y < src.rows; y++ ) {
        for( int x = 0; x < src.cols; x++ ) {
            for( int z = 0; z < 3; z++) {
                samples.at<float>(y + x*src.rows, z) = src.at<Vec3b>(y,x)[z];
            }
        }
    }

    kmeans(samples, k, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );

    dst = Mat(src.size(), src.type());
    for( int y = 0; y < src.rows; y++ ) {
        for( int x = 0; x < src.cols; x++ ) {
            int cluster_idx = labels.at<int>(y + x*src.rows,0);
            dst.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
            dst.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
            dst.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
        }
    }
}


Mat getShapeAsBinary(const Mat kmeans_image, int roi_width) {

    Vec3b color = kmeans_image.at<Vec3b>(roi_width/2,roi_width/2);
    cv::Mat bw_image;
    cv::inRange(kmeans_image, color, color, bw_image);

    std::vector<std::vector<Point> > contours;
    std::vector<Vec4i> hierarchy;

    //TODO: Test the hierarchy retriveval mode for optimal speed
    //TODO: Test contour approximation method
    cv::findContours(bw_image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    cv::drawContours(bw_image, contours,0, Scalar(255,255,255), CV_FILLED);

    return bw_image;
}
