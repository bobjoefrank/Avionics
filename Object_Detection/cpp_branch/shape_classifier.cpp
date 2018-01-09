#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "shape_classifier.h"

char const* classifyShape(std::vector<cv::Point> contours){
    /*
    std::vector<cv::Point> approx;
    cv::approxPolyDP(cv::Mat(canny_contours[i]), approx, cv::arcLength(cv::Mat(canny_contours[i]), true)*0.02, true);
    if(approx.size() == 4){
        std::cout << "its a square" << std::endl;
    }
    */
    //convexityDefects isContourConvex
    /*
    std::vector<cv::Point> approx;
    std::vector<cv::Point> hull;
    cv::approxPolyDP(cv::Mat(contours), approx, cv::arcLength(cv::Mat(contours), true)*0.02, true);
    if(approx.size() == 1){
        return "circle";
    }
    if(approx.size() == 2){
        return "semi_circle or quarter_circle";
    }
    if(approx.size() == 3){
        return "triangle";
    }
    if(approx.size() == 4){
        return "square rectangle or trapezoid";
    }
    if(approx.size() == 5){
        return "pentagon";
    }
    if(approx.size() == 6){
        return "hexagon";
    }
    if(approx.size() == 7){
        return "heptagon";
    }
    if(approx.size() == 8){
        return "octagon";
    }
    */
    
    int defects_counter;
    int min_defect_size = 5;
    std::vector<cv::Vec4i> defects;
    cv::convexHull(cv::Mat(contours), hull, false);
    if(hull.size() > 3){
        convexityDefects(contours, hull, defects);
    }

    return "can not determine";
}

std::vector<cv::Point> getMaxContour(cv::Mat& img, int orientation_min_area, int orientation_max_area)
{
    cv::Mat img_grey;
    cvtColor(img, img_grey, cv::COLOR_BGR2GRAY);

    cv::Mat img_binary;
    cv::threshold(img_grey, img_binary, 50, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    // Find all the contours in the thresholded image
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours;
    findContours(img_binary, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    //find max sized object within specified area limits
    int max_index = 0;
    int max_area = 0;
    for (size_t i = 0; i < contours.size(); ++i)
    {
        // Calculate the area of each contour
        double area = contourArea(contours[i]);
        // Ignore contours that are too small or too large
        if (area < orientation_min_area || orientation_max_area < area) continue;

        if(area > max_area){
            max_index = i;
            max_area = area;
        }
    }
    // Draw contour only for visualisation purposes
    //drawContours(img, contours[max_index], static_cast<int>(i), Scalar(0, 0, 255), 1, 8, hierarchy, 0);

    return contours[max_index];
}
