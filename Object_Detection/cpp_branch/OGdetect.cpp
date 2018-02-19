#include <stdio.h>
#include <iostream>
#include <cmath>
#include <Python.h>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

#include "orientation.h"
#include "classifier.h"
#include "objectdetection_functions.h"

void readme();
/* @function main */
int main( int argc, char** argv )
{

    //ratio to resize image before putting it in surf detection and then resizing keypoints
    float surf_resize_factor = 6;

    //number of ROI keypoints to compute with the greatest max_response
    int roi_count = 24;

    // how big cropped roi image will be
    float roi_width = 135;

    // padding color
    cv::Scalar padding_color = cvScalar(255,255,255);

    // kMeans parameters
    int k = 3;
    cv::Mat labels;
    int attempts = 30;
    cv::Mat centers;

    // keypoint grouping distance
    int keypoint_group_distance = 50;

    // Hessian value
    int minHessian = 1200;

    // canny edge detection threshold 0-100
    int canny_thresh = 0;
    // ratio for max threshold (suggested 3)
    int canny_ratio = 3;
    // kernel_sizes 3,5,7,9,11, makes a matrix of that size for canny edge detection
    int kernel_size = 3;

    // size limits for object when finding orientation
    int orientation_min_area = 0;
    int orientation_max_area = 1e5;
    // scales for visualization purposes only, adjust to lengthen lines
    // scale1 = green line, used as reference for radians
    // scale2 = blue line
    int line_size_1 = 14;
    int line_size_2 = 9;

    // shape classifier min and max area for contours
    //shape_min = 0;
    //shape_max = 1e5;



    // read image
    if( argc != 2 ){
        readme();
        return -1;
    }

    cv::Mat img = imread (argv[1]);
    cv::Mat img_hsv;
    cvtColor(img, img_hsv, CV_BGR2HSV);

    if( !img.data ){
        std::cout<< " Usage ./<executable_name> <img_name> " << std::endl; return -1;
    }


    //resizes image, finds keypoints, then returns keypoints with same aspect ratio as original image
    cv::Mat hsv_channels[3];
    cv::split(img_hsv, hsv_channels);
    cv::Mat used_channel = hsv_channels[2];

    cv::resize(used_channel, used_channel, cv::Size(img_hsv.cols/surf_resize_factor,img_hsv.rows/surf_resize_factor));

    std::vector<cv::KeyPoint> keypoints = get_SURF_detected_keypoints(used_channel, minHessian);
    imshow("hsv_channel", used_channel);

    //Convert keypoints back to original dimensions
    for(unsigned long i=0; i<keypoints.size()-1; ++i){
        keypoints[i].pt.x *= surf_resize_factor;
        keypoints[i].pt.y *= surf_resize_factor;
    }

    // if keypoints is less than certain number min:5 just discard the picture

    //think about running kmeans then blurring and kmeans again to connect broken parts of shape
    //look at the cross picture for example

    // Takes keypoint with strongest response from each cluster within a certain distance
    // also gives back keypoint when no other keypoint in range
    std::vector<cv::KeyPoint> grouped_keypoints;
    grouped_keypoints = group_keypoints(keypoints, keypoint_group_distance);


    // find and delete duplicates
    grouped_keypoints = delete_duplicates(grouped_keypoints);

    // draw keypoints
    cv::Mat img_keypoints;
    cv::drawKeypoints( img, grouped_keypoints, img_keypoints, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DEFAULT );

    // show detected (drawn) keypoints
    imshow("Keypoints", img_keypoints );

    //initialize python integration
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }
    Py_SetProgramName(program);  /* optional but recommended */
    Py_Initialize();

    char window_name[80];
    float x,y;
    int counter = 0;
    for(std::vector<cv::KeyPoint>::const_iterator i = grouped_keypoints.begin(); i != grouped_keypoints.end(); i++){

        //take only top number ROI with highest response, may not have all the keypoints in the images
        if(counter >= roi_count){
            break;
        }

        //delete later
        if(counter <= 2){
            counter++;
            continue;
        } else {
            counter++;
        }

        //add some nice spacing kek
        std::cout << "\n----------------------------\n" << std::endl;

        x = (i->pt.x)-(roi_width/2);
        y = (i->pt.y)-(roi_width/2);
        std::cout << "Keypoint:" << counter <<" x: " << x << " y: " << y << std::endl;

        // if the ROI region that we are cropping out goes out of bounds, adds a border to it as padding
        cv::Mat roi_image = getPaddedROI(img, x, y, roi_width, roi_width, padding_color);

        //
        // run kMeans
        //

        cv::Mat centers;
        cv::Mat samples(roi_image.rows * roi_image.cols, 3, CV_32FC2);
        for( int y = 0; y < roi_image.rows; y++ ) {
            for( int x = 0; x < roi_image.cols; x++ ) {
                for( int z = 0; z < 3; z++) {
                    samples.at<float>(y + x*roi_image.rows, z) = roi_image.at<Vec3b>(y,x)[z];
                }
            }
        }

        kmeans(samples, k, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );

        cv::Mat roi_kmeans( roi_image.size(), roi_image.type() );
        for( int y = 0; y < roi_image.rows; y++ ) {
            for( int x = 0; x < roi_image.cols; x++ ) {
                int cluster_idx = labels.at<int>(y + x*roi_image.rows,0);
                roi_kmeans.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
                roi_kmeans.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
                roi_kmeans.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
            }
        }

        // kMeans window name creator
        sprintf(window_name, "kMeans_no.%d", counter);
        imshow( window_name, roi_kmeans);

        //
        // canny edge detection and findcontours
        //

        /// Reduce noise with a kernel 3x3
        //blur( roi_image, roi_image, Size(9,9) );

        cv::Mat roi_canny_edges;
        cv::Canny(roi_kmeans, roi_canny_edges, canny_thresh, canny_thresh*canny_ratio, kernel_size);

        // canny_edge window name creator
        sprintf(window_name, "Canny_Edge_no.%d", counter);
        //imshow(window_name, roi_canny_edges);

        // find contours
        std::vector<std::vector<cv::Point> > canny_contours;
        std::vector<Vec4i> canny_hierarchy;
        cv::findContours(roi_canny_edges, canny_contours, canny_hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0));

        //approximates polygonal curves with polygon of less vertices to smooth (lessen) distance between vertices
        for (size_t i = 0; i < canny_contours.size(); ++i)
        {
            std::vector<cv::Point> approx;
            cv::approxPolyDP(cv::Mat(canny_contours[i]), canny_contours[i], cv::arcLength(cv::Mat(canny_contours[i]), true)*0.001, true);

        }

        // draw canny_contours for visualization purposes only
        std::vector<std::vector<cv::Point> > ocr_contours;
        for (size_t i = 0; i < canny_contours.size(); ++i)
        {
            // Calculate the area of each contour
            double area = contourArea(canny_contours[i]);
            // Ignore contours that are too small or too large
            if (area < 30 || 1e4 < area) continue;

            //collect contours that are within the area bounds
            ocr_contours.push_back(canny_contours[i]);

            cv::drawContours(roi_kmeans, canny_contours, static_cast<int>(i), Scalar(0, 0, 255), 1, 8, canny_hierarchy, 0);
        }

        //get rid of images without any contours
        if( !ocr_contours.size() ){
            std::cout<< "No contours detected" << std::endl;
            continue;
        }

        //get rid of images with contours that are too small
        //checks if the largest contour is smaller than acceptable size
        int max_index = 0;
        int max_area = 0;
        for (size_t i = 0; i < ocr_contours.size(); ++i)
        {
            double area = contourArea(ocr_contours[i]);
            if(area > max_area){
                max_index = i;
                max_area = area;
            }
        }
        if( max_area < 30 ){
            continue;
        }

        //
        // classify letters
        //

        std::cout << "number contours: " << ocr_contours.size() << std::endl;

        cv::Mat ocr_image_resized = createOcrImage(ocr_contours, max_index, roi_kmeans);

        sprintf(window_name, "ocr_resized_no.%d", counter);
        //imshow( window_name, ocr_image_resized);

        imwrite("../pictures/saved_ocr.png", ocr_image_resized);

        //call python ocr model serving program
        // PyRun_SimpleString("import sys, os\nsys.path.append('.')\nfrom python_ocr import *\n"
        //                     "test()");


        //
        // find orientation of objects
        //

        std::vector<Point> max_orientation_contour = getMaxContour(roi_kmeans, orientation_min_area, orientation_max_area);
        // Find the orientation of each object
        double angle = getAngle(max_orientation_contour, roi_kmeans, line_size_1, line_size_2);
        if( angle ){
            std::cout << "radians: " << angle << std::endl;
        } else {
            std::cout << "No Object detected" << std::endl;
        }

        //
        // find color of kmeans center points
        //

        for (int i = 0; i < k; ++i){
            std::cout << " R: " << (int)centers.at<float>(i,2) << " G: " << (int)centers.at<float>(i,1) << " B: " << (int)centers.at<float>(i,0) << std::endl;
            std::string color = findColor(centers, i);
            std::cout << color << std::endl;
        }

        sprintf(window_name, "kMeans_no.%d", counter);
        imshow( window_name, roi_kmeans );
    }

    //close python applications
    Py_Finalize();
    PyMem_RawFree(program);

    // wait for 'q' key to close
    char key = cv::waitKey(0);
    while ( key != 'q'){
        key = cv::waitKey(0);
    }
}

/* @function readme */
void readme()
{ std::cout << " Usage: ./<executable_name> <img_name>" << std::endl; }
