#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "shape_classifier.h"

cv::Mat createOcrImage(std::vector<std::vector<cv::Point> > ocr_contours, int max_index, cv::Mat img){
    //order contours from largest to smallest maybe?

    //fill in max contour which will be the outermost outline of the letter
    cv::Mat ocr_image = cv::Mat::zeros(img.size(), CV_8UC3);
    cv::fillConvexPoly( ocr_image, ocr_contours[max_index], cv::Scalar(255, 255, 255));

    //fill in the other contours which may or may not be the holes of the letter with random colors
    cv::RNG rng(12345);
    for (size_t  i = 0; i < ocr_contours.size(); ++i){
        if( i != max_index){
            cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
            cv::fillConvexPoly( ocr_image, ocr_contours[i], color);
        }
    }

    //crop out bounded box of letter
    cv::Rect bounding_box = boundingRect(ocr_contours[max_index]);
    cv::Mat ocr_cropped = ocr_image(bounding_box);

    //create new image with some space between edge of picture
    cv::Mat ocr_image_resized(ocr_cropped.size().height*1.3, ocr_cropped.size().width*1.3, CV_8UC3, cv::Scalar(0, 0, 0));

    //find center of new created image
    cv::Point ocr_center(ocr_image_resized.cols/2, ocr_image_resized.rows/2);

    //compute the rectangle centered in the image, same size as box
    cv::Rect center_box(ocr_center.x - bounding_box.width/2, ocr_center.y - bounding_box.height/2, bounding_box.width, bounding_box.height);

    //copy the letter into the resized image
    ocr_cropped.copyTo(ocr_image_resized(center_box));

    //resize image to 28x28
    cv::resize(ocr_image_resized, ocr_image_resized, cv::Size(28,28));

    return ocr_image_resized;
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
