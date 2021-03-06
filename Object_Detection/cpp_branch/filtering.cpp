#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

#include "filtering.h"

cv::Mat getPaddedROI(const cv::Mat &input, int top_left_x, int top_left_y, int width, int height, cv::Scalar paddingColor){
    int bottom_right_x = top_left_x + width;
    int bottom_right_y = top_left_y + height;

    cv::Mat output;
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

        cv::Rect R(top_left_x, top_left_y, width, height);
        cv::copyMakeBorder(input(R), output, border_top, border_bottom, border_left, border_right, cv::BORDER_CONSTANT, paddingColor);
    }
    else {
        // no border padding required
        cv::Rect R(top_left_x, top_left_y, width, height);
        output = input(R);
    }
    return output;
}

std::vector<cv::KeyPoint> GroupKeypoints(std::vector<cv::KeyPoint> keypoints, int keypoint_group_distance){
    int index_a = 0;
    for(std::vector<cv::KeyPoint>::const_iterator point_a = keypoints.begin(); point_a != keypoints.end(); point_a++){
        int index_b = 0;
        for(std::vector<cv::KeyPoint>::const_iterator point_b = keypoints.begin(); point_b != keypoints.end(); point_b++){
            if (std::abs(point_a->pt.x - point_b->pt.x) <= keypoint_group_distance && std::abs(point_a->pt.y - point_b->pt.y) <= keypoint_group_distance){
                if (point_b->response > point_a->response){
                    keypoints[index_a] = keypoints[index_b];
                }
            }
            index_b += 1;
        }
        index_a += 1;
    }
    
    // alternate cluster keypoint without stranded keypoints
    /*std::vector<cv::KeyPoint> keypoints;
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
        keypoints.push_back(keypoints[max_index]);
        index_a += 1;
    }*/

    return keypoints;
}

std::vector<cv::KeyPoint> DeleteDupes(std::vector<cv::KeyPoint> grouped_keypoints){
    unsigned long j = 0;
    while(j < grouped_keypoints.size()-1){
        for(size_t i= j+1 ; i < grouped_keypoints.size();){
            if( grouped_keypoints[j].pt.x == grouped_keypoints[i].pt.x ){
                grouped_keypoints.erase(grouped_keypoints.begin()+i);
            } else {
                ++i;
            }
        }
        ++j;
    }
    return grouped_keypoints;
}