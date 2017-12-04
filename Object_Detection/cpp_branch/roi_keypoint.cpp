#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

#include "padded_roi.h"
#include "orientation.h"

void readme();
/* @function main */
int main( int argc, char** argv )
{

    //how big roi radius will be
    float roi_width = 37;

    //padding color
    cv::Scalar padding_color = cvScalar(255,255,255);

    //kNN parameters
    int k = 3;
    cv::Mat labels;
    int attempts = 30;
    cv::Mat centers;

    //keypoint grouping distance
    int keypoint_group_distance = 40;

    //Hessian value
    int minHessian = 4500;

    //scales for visualization purposes only, adjust to lengthen lines
    //scale1 = green line, used as reference for radians
    //scale2 = blue line
    int scale1 = 10;
    int scale2 = 6;

    //canny edge detection threshold 0-100
    int canny_thresh = 5;
    //ratio for max threshold (suggested 3)
    int canny_ratio = 3;
    //kernel_sizes 3,5,7,9,11, makes a matrix of that size for canny edge detection
    int kernel_size = 3;




    //read image
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
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );
    std::vector<cv::KeyPoint> keypoints;
    detector->detect( img_grey, keypoints);


    // fix this loop, it gives repeats within distances, the pushback is wrong, change looping iterator to constants and use indexes
    // to get correct keypoints
    // Takes keypoint with strongest response from each cluster within a certain
    std::vector<cv::KeyPoint> grouped_keypoints;
    int index_a = 0;
    int max_index = 0;
    for(std::vector<cv::KeyPoint>::const_iterator point_a = keypoints.begin(); point_a != keypoints.end(); point_a++){
        int index_b = 0;
        float max_response = 0;
        for(std::vector<cv::KeyPoint>::const_iterator point_b = keypoints.begin(); point_b != keypoints.end(); point_b++){
            if (abs(point_a->pt.x - point_b->pt.x) <= keypoint_group_distance && abs(point_a->pt.y - point_b->pt.y) <= keypoint_group_distance){
                if (point_b->response > max_response){
                    max_index = index_b;
                }
            }
            index_b += 1;
        }
        grouped_keypoints.push_back(keypoints[max_index]);
        index_a += 1;
    }
    //find and delete duplicates
    int j = 0;
    while(j < grouped_keypoints.size()-1){
        for(int i= j+1 ; i < grouped_keypoints.size();){
            if( grouped_keypoints[j].pt.x == grouped_keypoints[i].pt.x ){
                grouped_keypoints.erase(grouped_keypoints.begin()+i);
            } else {
                ++i;
            }
        }
        ++j;
    }

    /*for(int i=0 ; i < grouped_keypoints.size()-1;){
        if ( grouped_keypoints[i].pt.x == grouped_keypoints[i+1].pt.x ){
            grouped_keypoints.erase(grouped_keypoints.begin()+i+1);
        } else {
            ++i;
        }
    }*/

    //draw keypoints
    cv::Mat img_keypoints;
    cv::drawKeypoints( img_grey, grouped_keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );

    //show detected (drawn) keypoints
    imshow("Keypoints", img_keypoints );

    char window_name[80];
    float x,y;
    int counter = 0;
    for(std::vector<cv::KeyPoint>::const_iterator i = grouped_keypoints.begin(); i != grouped_keypoints.end(); i++){
        counter++;
        x = (i->pt.x)-(roi_width/2);
        y = (i->pt.y)-(roi_width/2);
        std::cout << "x: " << x << " y: " << y << std::endl;

        //if the ROI region that we are cropping out goes out of bounds, adds a boder to it as padding
        cv::Mat roi_image = getPaddedROI(img, x, y, roi_width, roi_width, padding_color);
        //sprintf(window_name, "roi_no.%d", counter);
        //imshow( window_name, roi_image );

        //
        //run kMeans
        //
        cv::Mat centers;
        cv::Mat samples(roi_image.rows * roi_image.cols, 3, CV_32F);
            for( int y = 0; y < roi_image.rows; y++ )
                for( int x = 0; x < roi_image.cols; x++ )
                    for( int z = 0; z < 3; z++)
                        samples.at<float>(y + x*roi_image.rows, z) = roi_image.at<Vec3b>(y,x)[z];

        kmeans(samples, k, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );

        cv::Mat roi_kmeans( roi_image.size(), roi_image.type() );
        for( int y = 0; y < roi_image.rows; y++ ){
            for( int x = 0; x < roi_image.cols; x++ )
            {
                int cluster_idx = labels.at<int>(y + x*roi_image.rows,0);
                roi_kmeans.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
                roi_kmeans.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
                roi_kmeans.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
            }
        }

        //kMeans window name creator
        sprintf(window_name, "kMeans_no.%d", counter);
        imshow( window_name, roi_kmeans );


        //
        // find orientation of objects
        //
        cv::Mat roi_grey;
        //use roi_kmeans or roi_image??
        cvtColor(roi_image, roi_grey, COLOR_BGR2GRAY);
        Mat roi_binary;
        threshold(roi_grey, roi_binary, 50, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        // Find all the contours in the thresholded image
        vector<Vec4i> hierarchy;
        vector<vector<Point> > contours;
        findContours(roi_binary, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
        for (size_t i = 0; i < contours.size(); ++i)
        {
            // Calculate the area of each contour
            double area = contourArea(contours[i]);
            // Ignore contours that are too small or too large
            if (area < 1e2 || 1e5 < area) continue;

            // Draw each contour only for visualisation purposes
            //drawContours(roi_kmeans, contours, static_cast<int>(i), Scalar(0, 0, 255), 1, 8, hierarchy, 0);

            // Find the orientation of each shape
            double angle = getOrientation(contours[i], roi_kmeans, scale1, scale2);
            std::cout << "radians: " << angle << std::endl;
        }


        //
        //canny edge detection and findcontours
        //
        cv::Mat roi_grey_blur;
        cv::blur(roi_grey, roi_grey_blur, Size(1,1));

        cv::Mat roi_canny_edges;
        cv::Canny(roi_grey_blur, roi_canny_edges, canny_thresh, canny_thresh*canny_ratio, kernel_size);

        //canny_edge window name creator
        sprintf(window_name, "Canny_Edge_no.%d", counter);
        imshow(window_name, roi_canny_edges);

        //find contours
        std::vector<std::vector<Point> > canny_contours;
        std::vector<Vec4i> canny_hierarchy;
        cv::findContours(roi_canny_edges, canny_contours, canny_hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0));

        //draw canny_contours for visualization purposes only
        for (size_t i = 0; i < canny_contours.size(); ++i)
        {
            // Calculate the area of each contour
            double area = contourArea(canny_contours[i]);
            // Ignore contours that are too small or too large
            if (area < 4 || 1e5 < area) continue;
            cv::drawContours(roi_kmeans, canny_contours, static_cast<int>(i), Scalar(0, 0, 255), 1, 8, canny_hierarchy, 0);
        }
        sprintf(window_name, "kMeans_no.%d", counter);
        imshow( window_name, roi_kmeans );




        // was trying to find centerpoints but points were falling out of bounds

        //cvtColor(roi_kmeans, roi_kmeans, CV_BGR2HSV);
        //Point2f c = centers.at<Point2f>(0);
        //circle(roi_kmeans, c, 15, cv::Scalar::all(-1), 5, LINE_AA);
        //Point pt = c;
        //std::cout << pt.x << ", " << pt.y << std::endl;

    }

    imshow("Keypoints", img_keypoints );
    cv::waitKey(0);
    return 0;
}

/* @function readme */
void readme()
{ std::cout << " Usage: ./SURF_detector <img1>" << std::endl; }
