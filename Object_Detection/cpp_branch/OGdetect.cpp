#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/types_c.h>

#include <fdeep/fdeep.hpp>

#include "orientation.h"
#include "classifier.h"
#include "filtering.h"


// flood fill cornell

// find way to get contours first (blur image or resize), after gettting contour, get largest contour
// to crop the original image, then take that cropped image and run kmeans
// after kmeans, get contour of letter inside and crop
// this method also makes it easier to find the color by taking the cropped image
// and finding the color of that

// rotate only the letters and not the shapes

// think about if keypoints is less than certain number min:5 just discard the 

// take top number with responses above a certain value or a set number from the top
// whichever comes first, but more likely those above a certain response

// 96 ft x 64 ft at 200 ft altitude
// 4.8 pixels per inch density

// cornell false positive, if confidence of letter and shape low thne discard image
// if color of shape and leter are the same, then discard

// SEPERATE BY KMEANS COLORS AND FIND CONTOURS AND RUN OCR FOR EACH COLOR


void readme();

int main( int argc, char** argv )
{

    //ratio to resize image before putting it in surf detection and then resizing keypoints
    float surf_resize_factor = 8;

    //number of ROI keypoints to compute with the greatest max_response
    int roi_count = 20;

    // how big cropped roi image will be, total width
    float roi_width = 230;

    // padding color
    cv::Scalar padding_color = cv::Scalar(255,255,255);

    // kMeans parameters
    int k = 3;
    int attempts = 10;

    // keypoint grouping distance
    int keypoint_group_distance = 400;

    // SURF hessian value
    int minHessian = 500;

    // canny edge detection threshold 0-100
    int canny_thresh = 0;
    // ratio for max threshold (suggested 3)
    int canny_ratio = 3;
    // kernel_sizes 3,5,7 makes a matrix of that size for canny edge detection
    int kernel_size = 3;

    // i think 5 was best change this back to 5 or 7 when done testing

    // scales for visualization purposes only, adjust to lengthen lines
    // scale1 = green line, used as reference for radians
    // scale2 = blue line
    int line_size_1 = 14;
    int line_size_2 = 9;

    // letter classifier min and max area for contours
    int letter_min_area = 100;
    int letter_max_area = 1e9;

    // read image
    if( argc < 2 ){
        readme();
        return -1;
    }

    cv::Mat img = cv::imread (argv[1]);
    if( !img.data ){
        readme();
        return -1;
     }

    // convert to hsv color space for SURF detection and kmeans
    cv::Mat img_hsv;
    cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);
    //resizes image, finds keypoints, then returns keypoints with same aspect ratio as original image
    cv::Mat img_hsv_resized;
    cv::resize(img_hsv, img_hsv_resized, cv::Size(img_hsv.cols/surf_resize_factor,img_hsv.rows/surf_resize_factor));
    cv::Mat img_hsv_resized_blurred;
    cv::GaussianBlur( img_hsv_resized, img_hsv_resized_blurred, cv::Size(3,3), 0, 0);
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );
    std::vector<cv::KeyPoint> keypoints;
    detector->detect( img_hsv_resized_blurred, keypoints);

    // save keypoints for grey_blurred image
    std::vector<cv::KeyPoint>resized_keypoints = keypoints;

    // convert keypoints back to original dimensions
    for(unsigned long i=0; i<keypoints.size()-1; ++i){
        keypoints[i].pt.x *= surf_resize_factor;
        keypoints[i].pt.y *= surf_resize_factor;
    }

    for(std::vector<cv::KeyPoint>::const_iterator point_a = keypoints.begin(); point_a != keypoints.end(); point_a++){
        std::cout << point_a->response << std::endl;
    }

    // Takes keypoint with strongest response from each cluster within a certain distance
    // also gives back keypoint when no other keypoints in range
    std::vector<cv::KeyPoint> grouped_keypoints = GroupKeypoints(keypoints, keypoint_group_distance);

    // find and delete duplicates
    grouped_keypoints = DeleteDupes(grouped_keypoints);

    std::cout<< "Number of keypoints: " << grouped_keypoints.size() << std::endl;

    // draw keypoints
    cv::Mat img_keypoints;
    cv::drawKeypoints(img, grouped_keypoints, img_keypoints, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DEFAULT );
    cv::drawKeypoints(img_hsv_resized_blurred, resized_keypoints, img_hsv_resized_blurred, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DEFAULT );

    // show detected (drawn) keypoints
    //imshow("Keypoints", img_keypoints );
    imshow("img_hsv_resized_blurred", img_hsv_resized_blurred);

    //initialize fdeep model
    const auto model = fdeep::load_model("fdeep_model.json");

    char window_name[80];
    float x,y;
    int counter = 0;
    for(std::vector<cv::KeyPoint>::const_iterator i = grouped_keypoints.begin(); i != grouped_keypoints.end(); i++){

        //take only top number ROI with highest response, may not have all the keypoints in the images
        if(counter >= roi_count){
            break;
        }
        counter++;

        //add some nice spacing kek
        std::cout << "\n----------------------------\n" << std::endl;

        x = (i->pt.x)-(roi_width/2);
        y = (i->pt.y)-(roi_width/2);
        std::cout << "Keypoint:" << counter <<" x: " << x << " y: " << y << std::endl;

        // if the ROI region that we are cropping out goes out of bounds, adds a border to it as padding
        cv::Mat roi_image = getPaddedROI(img_hsv, x, y, roi_width, roi_width, padding_color);

        //
        // run kMeans
        //

        // put in new file called filtering and find out how to return two types, centers and mat image
        cv::Mat labels;
        cv::Mat centers;
        cv::Mat samples(roi_image.rows * roi_image.cols, 3, CV_32F);
        for( int y = 0; y < roi_image.rows; y++ ) {
            for( int x = 0; x < roi_image.cols; x++ ) {
                for( int z = 0; z < 3; z++) {
                    samples.at<float>(y + x*roi_image.rows, z) = roi_image.at<cv::Vec3b>(y,x)[z];
                }
            }
        }

        kmeans(samples, k, labels, cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, cv::KMEANS_PP_CENTERS, centers );

        cv::Mat roi_kmeans( roi_image.size(), roi_image.type() );
        for( int y = 0; y < roi_image.rows; y++ ) {
            for( int x = 0; x < roi_image.cols; x++ ) {
                int cluster_idx = labels.at<int>(y + x*roi_image.rows,0);
                roi_kmeans.at<cv::Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
                roi_kmeans.at<cv::Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
                roi_kmeans.at<cv::Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
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

        std::vector<std::vector<cv::Point> > canny_contours_ccomp;
        std::vector<cv::Vec4i> canny_hierarchy_ccomp;
        cv::findContours(roi_canny_edges, canny_contours_ccomp, canny_hierarchy_ccomp, cv::RetrievalModes::RETR_CCOMP, cv::ContourApproximationModes::CHAIN_APPROX_NONE, cv::Point(0,0));

        //approximates polygonal curves with polygon of less vertices to smooth (lessen) distance between vertices
        for (size_t i = 0; i < canny_contours_ccomp.size(); ++i)
        {
            std::vector<cv::Point> approx;
            cv::approxPolyDP(cv::Mat(canny_contours_ccomp[i]), canny_contours_ccomp[i], cv::arcLength(cv::Mat(canny_contours_ccomp[i]), true)*0.001, true);
        }

        //get rid of images without any contours
        if( !canny_contours_ccomp.size() ){
            std::cout<< "No contours detected" << std::endl;
            continue;
        }

        //checks if the largest contour is smaller than acceptable size
        int max_index = 0;
        int max_area = 0;
        int large_contour_num = 0;
        for (size_t i = 0; i < canny_contours_ccomp.size(); ++i)
        {
            // calculate the area of each contour
            double area = contourArea(canny_contours_ccomp[i]);
            if(counter == 5 || counter == 6){
                std::cout << area << " counter number: " << (int)i << std::endl;
            }
            if(area > max_area){
                max_index = i;
                max_area = area;
            }
            //ignore contours too small
            if (area < letter_min_area || area > letter_max_area) continue;

            //draw for visuals only if contours big enough
            large_contour_num++;
            cv::drawContours(roi_kmeans, canny_contours_ccomp, static_cast<int>(i), cvScalar(255, 255, 255), 1, 8, canny_hierarchy_ccomp, 0);
        }
        if( max_area < letter_min_area ){
            std::cout<< "Contours too small" << std::endl;
            continue;
        }

        //
        // classify letters
        //

        if(counter == 5 || counter == 6){
            std::cout << "number contours: " << large_contour_num << std::endl;
            cv::Mat ocr_cropped = CropOcrImage(canny_contours_ccomp, max_index, roi_kmeans, canny_hierarchy_ccomp, letter_min_area,false);
            cv::Mat ocr_resized = ResizeOcrImage(ocr_cropped, false);

            cv::Mat ocr_rotated_45 = RotateOcrImage45(ocr_cropped);
            cv::Mat ocr_resized_45 = ResizeOcrImage(ocr_rotated_45, true);
            
            // invert with white mask to find inner letter/object
            cv::Mat ocr_cropped_invert = CropOcrImage(canny_contours_ccomp, max_index, roi_kmeans, canny_hierarchy_ccomp, letter_min_area, true);

            // canny edge to find the new edges of the inverted image
            



            cv::Mat ocr_resized_invert = ResizeOcrImage(ocr_cropped_invert, false);

            cv::Mat ocr_rotated_invert_45 = RotateOcrImage45(ocr_cropped_invert);
            cv::Mat ocr_resized_invert_45 = ResizeOcrImage(ocr_rotated_invert_45, true);

            cv::Mat test_img;
            cv::Mat white_mask(roi_kmeans.rows, roi_kmeans.cols, roi_kmeans.type(), cv::Scalar::all(255));

            int degrees1 = 0;
            int degrees2 = 45;
            for(int i = 0; i < 4; ++i){
                sprintf(window_name, "ocr_no.%d_%d", counter, degrees1);
                //imshow( window_name, ocr_resized);
                if (degrees1 == 0){
                    imshow( window_name, ocr_resized_invert);
                    cv::cvtColor(ocr_resized, test_img, cv::COLOR_BGR2GRAY);
                }
                cv::rotate(ocr_resized, ocr_resized, 0);

                sprintf(window_name, "ocr_no.%d_%d", counter, degrees2);
                //imshow( window_name, ocr_resized_45);
                cv::rotate(ocr_resized_45, ocr_resized_45, 0);

                degrees1 += 90;
                degrees2 += 90;
            }
            std::map<int, int> mappings = {
                {0, 48}, {1, 49}, {2, 50}, {3, 51}, {4, 52}, {5, 53}, {6, 54}, {7, 55}, 
                {8, 56}, {9, 57}, {10, 65}, {11, 66}, {12, 67}, {13, 68}, {14, 69}, {15, 70}, 
                {16, 71}, {17, 72}, {18, 73}, {19, 74}, {20, 75}, {21, 76}, {22, 77}, {23, 78}, 
                {24, 79}, {25, 80}, {26, 81}, {27, 82}, {28, 83}, {29, 84}, {30, 85}, {31, 86}, 
                {32, 87}, {33, 88}, {34, 89}, {35, 90}, {36, 97}, {37, 98}, {38, 99}, {39, 100}, 
                {40, 101}, {41, 102}, {42, 103}, {43, 104}, {44, 105}, {45, 106}, {46, 107}, 
                {47, 108}, {48, 109}, {49, 110}, {50, 111}, {51, 112}, {52, 113}, {53, 114}, 
                {54, 115}, {55, 116}, {56, 117}, {57, 118}, {58, 119}, {59, 120}, {60, 121}, {61, 122}
            };
            const auto input = fdeep::tensor3_from_bytes(test_img.ptr(), test_img.rows, test_img.cols, test_img.channels(), 0.0f, 1.0f);
            const auto result = model.predict({input});
            auto prediction = fdeep::internal::tensor3_max_pos(result.front()).z_;
            std::cout << "Prediction: " << (char)(mappings[prediction]) << std::endl;
        }

        //
        // find orientation of objects
        //

        // Find the orientation of each object
        double angle = getAngle(canny_contours_ccomp[max_index], roi_kmeans, line_size_1, line_size_2);
        if( angle ){
            std::cout << "radians: " << angle << std::endl;
        } else {
            std::cout << "No Object detected" << std::endl;
        }

        //
        // find color of kmeans center points
        //

        for (int i = 0; i < 3; ++i){
            std::cout << " R: " << (int)centers.at<float>(i,2) << " G: " << (int)centers.at<float>(i,1) << " B: " << (int)centers.at<float>(i,0) << std::endl;
            // std::string color = findColor(centers, i);
            // std::cout << color << std::endl;
        }

        sprintf(window_name, "kMeans_no.%d", counter);
        imshow( window_name, roi_kmeans );
    }

    // wait for 'q' key to close
    char key = cv::waitKey(0);
    while ( key != 'q'){
        key = cv::waitKey(0);
    }
}

/* @function readme */
void readme(){
    std::cout << " Usage: ./<executable_name> <img_name>" << std::endl; 
}
