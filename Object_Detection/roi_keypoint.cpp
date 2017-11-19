#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

#include "padded_roi.h"

void readme();
/* @function main */
int main( int argc, char** argv )
{

    //how big roi radius will be
    float roi_width = 100;

    //padding color
    cv::Scalar padding_color = cvScalar(255,255,255);

    if( argc != 2 ){
        readme(); return -1;
    }

    cv::Mat img = imread (argv[1]);
    cv::Mat img_grey = imread( argv[1], cv::IMREAD_GRAYSCALE );
    cv::Mat img_hsv;
    cvtColor(img, img_hsv, CV_BGR2HSV);

    if( !img.data ){
        std::cout<< " Usage ./<executable_name> <img_name> " << std::endl; return -1;
    }

    //detect the keypoints using SURF Detector
    int minHessian = 4000;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );
    std::vector<cv::KeyPoint> keypoints_1;
    detector->detect( img_grey, keypoints_1 );

    //draw keypoints
    cv::Mat img_keypoints_1;
    cv::drawKeypoints( img_grey, keypoints_1, img_keypoints_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );

    //show detected (drawn) keypoints
    imshow("Keypoints", img_keypoints_1 );

    char window_name[80];
    float x,y;
    int counter = 0;
    for(std::vector<cv::KeyPoint>::const_iterator i = keypoints_1.begin(); i != keypoints_1.end(); ++i){
        counter++;
        x = (i->pt.x)-(roi_width/2);
        y = (i->pt.y)-(roi_width/2);
        std::cout << "x: " << x << " y: " << y << std::endl;

        //name for each window
        sprintf(window_name, "ROIno.%d", counter);
        //if the ROI region that we are cropping out goes out of bounds, adds a boder to it as padding
        cv::Mat roi_image = getPaddedROI(img, x, y, roi_width, roi_width, padding_color);
        imshow(window_name, roi_image);

    }

    //imshow("img_hsv", img_hsv);

    cv::waitKey(0);
    return 0;
}
/* @function readme */
void readme()
{ std::cout << " Usage: ./SURF_detector <img1>" << std::endl; }
