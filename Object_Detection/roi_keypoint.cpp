#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"


void readme();
/* @function main */
int main( int argc, char** argv )
{
    if( argc != 2 )
    { readme(); return -1; }
    cv::Mat img_1 = imread( argv[1], cv::IMREAD_GRAYSCALE );

    if( !img_1.data )
    { std::cout<< " --(!) Error reading image " << std::endl; return -1; }
    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 4000;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );
    std::vector<cv::KeyPoint> keypoints_1;
    detector->detect( img_1, keypoints_1 );

    //-- Draw keypoints
    cv::Mat img_keypoints_1;
    cv::drawKeypoints( img_1, keypoints_1, img_keypoints_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );

    //-- Show detected (drawn) keypoints
    imshow("Keypoints 1", img_keypoints_1 );

    //std::cout<< keypoints_1 << std::endl;

    cv::waitKey(0);
    return 0;
}
/* @function readme */
void readme()
{ std::cout << " Usage: ./SURF_detector <img1>" << std::endl; }
