#include <stdio.h>
#include <iostream>
#include <Python.h>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

#include "padded_roi.h"
#include "orientation.h"
#include "shape_classifier.h"

void readme();
/* @function main */
int main( int argc, char** argv )
{

    //number of ROI keypoints to compute with the greatest max_response
    int roi_count = 40;

    // how big roi radius will be
    float roi_width = 95;

    // padding color
    cv::Scalar padding_color = cvScalar(255,255,255);

    // kMeans parameters
    int k = 4;
    cv::Mat labels;
    int attempts = 30;
    cv::Mat centers;

    // keypoint grouping distance
    int keypoint_group_distance = 50;

    // Hessian value
    int minHessian = 6000;

    // canny edge detection threshold 0-100
    int canny_thresh = 5;
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
    cv::Mat img_grey = imread( argv[1], cv::IMREAD_GRAYSCALE );
    cv::Mat img_hsv;
    cvtColor(img, img_hsv, CV_BGR2HSV);

    if( !img.data ){
        std::cout<< " Usage ./<executable_name> <img_directory> " << std::endl; return -1;
    }

    // detect the keypoints using SURF Detector
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );
    std::vector<cv::KeyPoint> keypoints;
    detector->detect( img_grey, keypoints);

    // if keypoints is less than certain number min:5 just discard the picture

    // binarize kmeans photo and then run edge detection

    //use cv chain approx simple and then create the shape from that

    //think about running kmeans then blurring and kmeans again to connect broken parts of shape
    //look at the cross picture for example


    // Takes keypoint with strongest response from each cluster within a certain distance
    // also gives back keypoint when no other keypoint in range
    std::vector<cv::KeyPoint> grouped_keypoints;
    grouped_keypoints = keypoints;
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

    // find and delete duplicates
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
        if(counter <= 35){
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
        //sprintf(window_name, "roi_no.%d", counter);
        //imshow( window_name, roi_image );

        //
        // run kMeans
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

        // kMeans window name creator
        sprintf(window_name, "kMeans_no.%d", counter);
        imshow( window_name, roi_kmeans );

        //
        // canny edge detection and findcontours
        //

        /// Reduce noise with a kernel 3x3
        //blur( roi_kmeans, roi_kmeans, Size(3,3) );

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
            cv::approxPolyDP(cv::Mat(canny_contours[i]), canny_contours[i], cv::arcLength(cv::Mat(canny_contours[i]), true)*0.0025, true);

        }

        // draw canny_contours for visualization purposes only
        std::vector<std::vector<cv::Point> > ocr_contours;
        for (size_t i = 0; i < canny_contours.size(); ++i)
        {
            // Calculate the area of each contour
            double area = contourArea(canny_contours[i]);
            // Ignore contours that are too small or too large
            if (area < 450 || 1e5 < area) continue;

            //collect contours that are within the area bounds
            ocr_contours.push_back(canny_contours[i]);

            cv::drawContours(roi_kmeans, canny_contours, static_cast<int>(i), Scalar(0, 0, 255), 1, 8, canny_hierarchy, 0);
        }
        sprintf(window_name, "kMeans_no.%d", counter);
        imshow( window_name, roi_kmeans );


        //
        // classify shapes
        //

        // convert to OCR compatible image
        if(counter == 37)
        {
            std::cout << "number contours: " << ocr_contours.size() << std::endl;

            //order contours from largest to smallest
            //code here

            //find max sized contour
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

            //fill in max contour which will be the outermost outline of the letter
            cv::Mat ocr_image = cv::Mat::zeros(roi_kmeans.size(), CV_8UC3);
            cv::fillConvexPoly( ocr_image, ocr_contours[max_index], cv::Scalar(255, 255, 255));

            cv::fillConvexPoly( ocr_image, ocr_contours[3], cv::Scalar(0, 0, 0));
            cv::fillConvexPoly( ocr_image, ocr_contours[2], cv::Scalar(0, 0, 0));
            cv::fillConvexPoly( ocr_image, ocr_contours[4], cv::Scalar(0, 0, 0));

            /*
            //fill in the other contours which may or may not be the holes of the letter with black space
            RNG rng(12345);
            for (size_t i = 0; i < ocr_contours.size(); ++i){
                if( i != max_index){
                    cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
                    cv::fillConvexPoly( ocr_image, ocr_contours[i], color);
                }
            }
            */

            imshow( "ocr_image", ocr_image);

            //crop out bounded box of letter
            cv::Rect bounding_box = boundingRect(ocr_contours[max_index]);
            cv::Mat ocr_cropped = ocr_image(bounding_box);
            imshow( "ocr_cropped", ocr_cropped);

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
            imshow( "ocr_image_resized", ocr_image_resized);

            imwrite("../pictures/saved_ocr.png", ocr_image_resized);

            //call python ocr model serving program
            PyRun_SimpleString("import sys, os\nsys.path.append('.')\nfrom python_ocr import *\n"
                                "test()");

        }

        //cv::Mat roi_shapes;
        //drawContours(roi_kmeans, canny_contours, -1, Scalar(255), CV_FILLED, 8);
        //std::vector<Point> max_shape_contour = getMaxContour(roi_kmeans, shape_min, shape_max);
        // for (size_t i = 0; i < canny_contours.size(); ++i)
        // {
        //     // Calculate the area of each contour
        //     double area = contourArea(canny_contours[i]);
        //     // Ignore contours that are too small or too large
        //     if (area < 300 || 1e5 < area) continue;
        //
        //     std::cout << "shape: " << classifyShape(canny_contours[i]) << std::endl;
        //     /*
        //     // drawing the convex contours for testing only
        //     std::vector<cv::Point> hull;
        //     std::vector<int> hull_int;
        //     std::vector<cv::Vec4i> defects;
        //     cv::convexHull(canny_contours[i], hull, false);
        //     cv::convexHull(canny_contours[i], hull_int, false);
        //     if(hull_int.size() > 3){
        //         convexityDefects(canny_contours[i], hull_int, defects);
        //     }
        //     int convex_counter =0;
        //     for(int j=0; j<defects.size(); ++j){
        //         const cv::Vec4i& v = defects[j];
        //         float depth = v[3] / 256;
        //         if(depth > 5){
        //             int startidx = v[0]; Point ptStart(canny_contours[i][startidx]);
        //             int endidx = v[1]; Point ptEnd(canny_contours[i][endidx]);
        //             int faridx = v[2]; Point ptFar(canny_contours[i][faridx]);
        //
        //             line(roi_kmeans, ptStart, ptEnd, Scalar(0, 255, 0), 1);
        //             line(roi_kmeans, ptStart, ptFar, Scalar(0, 255, 0), 1);
        //             line(roi_kmeans, ptEnd, ptFar, Scalar(0, 255, 0), 1);
        //             //circle(roi_kmeans, ptFar, 4, Scalar(0, 255, 0), 2);
        //             convex_counter++;
        //         }
        //     }
        //     std::cout << convex_counter << std::endl;
        //     */
        // }






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
        imshow( window_name, roi_kmeans );


        // was trying to find centerpoints but points were falling out of bounds
        // center points would be the points used for finding whichthe kmeans group color is

        //cvtColor(roi_kmeans, roi_kmeans, CV_BGR2HSV);
        //Point2f c = centers.at<Point2f>(0);
        //circle(roi_kmeans, c, 15, cv::Scalar::all(-1), 5, LINE_AA);
        //Point pt = c;
        //std::cout << pt.x << ", " << pt.y << std::endl;
    }

    //close python applications
    Py_Finalize();
    PyMem_RawFree(program);

    imshow("Keypoints", img_keypoints );
    cv::waitKey(0);
    return 0;
}

/* @function readme */
void readme()
{ std::cout << " Usage: ./<executable_name> <img_directory>" << std::endl; }
