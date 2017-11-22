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
    float roi_width = 50;

    //padding color
    cv::Scalar padding_color = cvScalar(255,255,255);

    //kNN parameters
    int k = 3;
    cv::Mat labels;
    int attempts = 30;
    cv::Mat centers;

    if( argc != 2 ){
        readme();
        return -1;
    }

    cv::Mat img = imread (argv[1]);
    cv::Mat img_grey = imread( argv[1], cv::IMREAD_GRAYSCALE );
    cv::Mat img_hsv;
    cvtColor(img, img_hsv, CV_BGR2HSV);

    if( !img.data ){
        std::cout<< " Usage ./<executable_name> <img_name> " << std::endl; return -1;
    }

    //detect the keypoints using SURF Detector
    int minHessian = 4500;
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

        //name for each window for ROI
        sprintf(window_name, "ROIno.%d", counter);
        //if the ROI region that we are cropping out goes out of bounds, adds a boder to it as padding
        cv::Mat roi_image = getPaddedROI(img, x, y, roi_width, roi_width, padding_color);
        //imshow(window_name, roi_image);

        //hsv conversion before using kNN
        sprintf(window_name, "HSVno.%d", counter);
        //cvtColor(roi_image, roi_image, CV_BGR2HSV);
        //imshow(window_name, roi_image);

        //kNN
        sprintf(window_name, "kNNno.%d", counter);
        cv::Mat samples(roi_image.rows * roi_image.cols, 3, CV_32F);
            for( int y = 0; y < roi_image.rows; y++ )
                for( int x = 0; x < roi_image.cols; x++ )
                    for( int z = 0; z < 3; z++)
                        samples.at<float>(y + x*roi_image.rows, z) = roi_image.at<Vec3b>(y,x)[z];

        kmeans(samples, k, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );

        Mat kmeans_image( roi_image.size(), roi_image.type() );
        for( int y = 0; y < roi_image.rows; y++ )
            for( int x = 0; x < roi_image.cols; x++ )
            {
                int cluster_idx = labels.at<int>(y + x*roi_image.rows,0);
                kmeans_image.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
                kmeans_image.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
                kmeans_image.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
            }
            //cvtColor(new_image, new_image, CV_HSV2BGR);
            imshow( window_name, kmeans_image );
            //std::cout << centers << std::endl;
        }

    cv::waitKey(0);
    return 0;
}
/* @function readme */
void readme()
{ std::cout << " Usage: ./SURF_detector <img1>" << std::endl; }
