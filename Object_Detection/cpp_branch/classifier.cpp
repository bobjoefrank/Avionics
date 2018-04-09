#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "classifier.h"

cv::Mat CropOcrImage(std::vector<std::vector<cv::Point> > ocr_contours, int max_index, cv::Mat img, std::vector<cv::Vec4i> hierarchy){
    //order contours from largest to smallest maybe?

    //fill in max contour which will be the outermost outline of the letter
    cv::Mat ocr_image = cv::Mat::zeros(img.size(), CV_8UC3);
    cv::fillConvexPoly( ocr_image, ocr_contours[max_index], cv::Scalar(255, 255, 255));

    //fill in the other contours which may or may not be the holes of the letter with random colors
    cv::RNG rng(12345);
    for (size_t  i = 0; i < ocr_contours.size(); ++i){
        if( (int)i != max_index){
            cv::Scalar color = cv::Scalar(0,0,0);
            if(hierarchy[i][3] == -1){
                cv::fillConvexPoly( ocr_image, ocr_contours[i], color);
            }
        }
    }

    //crop out bounded box of letter
    cv::Rect bounding_box = boundingRect(ocr_contours[max_index]);
    cv::Mat ocr_cropped = ocr_image(bounding_box);

    return ocr_cropped;
}

cv::Mat ResizeOcrImage(cv::Mat ocr_rotated, int rotated_45){
    //create new image with some space between edge of picture
    cv::Mat ocr_rotated_resized(ocr_rotated.size().height*1.3, ocr_rotated.size().width*1.3, CV_8UC3, cv::Scalar(0, 0, 0));
    //rotating by 45 degrees causes extra border so this will make the border smaller
    std::cout << ocr_rotated_resized.cols << std::endl;
    if(rotated_45){
        cv::resize(ocr_rotated_resized, ocr_rotated_resized, cv::Size(ocr_rotated_resized.rows/1.15,ocr_rotated_resized.cols/1.15));
    }

    //find center of new created image
    cv::Point ocr_center(ocr_rotated_resized.cols/2, ocr_rotated_resized.rows/2);

    //compute the rectangle centered in the image, same size as box
    cv::Rect center_box(ocr_center.x - ocr_rotated.cols/2, ocr_center.y - ocr_rotated.rows/2, ocr_rotated.cols, ocr_rotated.rows);

    //copy the letter into the resized image
    ocr_rotated.copyTo(ocr_rotated_resized(center_box));

    //resize image to 28x28
    cv::resize(ocr_rotated_resized, ocr_rotated_resized, cv::Size(28,28));

    return ocr_rotated_resized;
}

cv::Mat RotateOcrImage45(cv::Mat ocr_cropped){
    double angle = -45;

    // get rotation matrix for rotating the image around its center
    cv::Point2f center(ocr_cropped.cols/2.0, ocr_cropped.rows/2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    // determine bounding rectangle
    cv::Rect bbox = cv::RotatedRect(center,ocr_cropped.size(), angle).boundingRect();
    // adjust transformation matrix
    rot.at<double>(0,2) += bbox.width/2.0 - center.x;
    rot.at<double>(1,2) += bbox.height/2.0 - center.y;

    cv::Mat ocr_rotated;
    cv::warpAffine(ocr_cropped, ocr_rotated, rot, bbox.size());
    return ocr_rotated;
}

std::string findColor(cv::Mat centers, int index){
    double B = centers.at<float>(index,0);
    double G = centers.at<float>(index,1);
    double R = centers.at<float>(index,2);

    double Cmax, Cmin;
    Cmax = (B > G)? B : G;
    Cmax = (R > Cmax)? R : Cmax;
    Cmin = (B > G)? G : B;
    Cmin = (R > Cmin)? Cmin : R;

    double delta = Cmax - Cmin;

    double H;
    if(Cmax == R){
        H = 60 * (fmod(((G-B)/delta),6));
    } else if(Cmax == G){
        H = 60 * (((B-R)/delta)+2);
    } else if(Cmax == B){
        H = 60 * (((R-G)/delta)+4);
    } else {
        H = 0;
    }

    double S;
    if(Cmax >= 0.001){
        S = delta/Cmax;
    } else {
        S = 0;
    }

    double V;
    V = Cmax/255;

    //readable HSV values
    std::cout << " H: " << H <<" S: " << S << " V: " << V << std::endl;

    //opencv conversion
    H *= 0.5;
    S *= 255;
    V *= 255;

    // add colors grey and brown

    if(V<92){
        return "black";
    }
    if(S<70){
        if(V<230){
            return "grey";
        } else {
            return "white";
        }
    }
    if(H>168){
        return "red";
    }
    if(H>128){
        return "purple";
    }
    if(H>86){
        return "blue";
    }
    if(H>40){
        return "green";
    }
    if(H>27){
        if(V <= 91 && V >= 31){
            return "brown";
        } else {
            return "yellow";
        }
    }
    if(H>11){
        if(V <= 91 && V >= 31){
            return "brown";
        } else {
            return "orange";
        }
    }
    if(H>=0){
        return "red";
    }
    return "COLOR SCALE ERROR";
}

std::vector<cv::Point> getMaxContour(cv::Mat& img, int oRentation_min_area, int oRentation_max_area)
{
    cv::Mat img_grey;
    cvtColor(img, img_grey, cv::COLOR_BGR2GRAY);

    cv::Mat img_Bnary;
    cv::threshold(img_grey, img_Bnary, 50, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    // Find all the contours in the thresholded image
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours;
    findContours(img_Bnary, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    //find max sized object within specified area limits
    int max_index = 0;
    int max_area = 0;
    for (size_t i = 0; i < contours.size(); ++i)
    {
        // Calculate the area of each contour
        double area = contourArea(contours[i]);
        // Ignore contours that are too small or too large
        if (area < oRentation_min_area || oRentation_max_area < area) continue;

        if(area > max_area){
            max_index = i;
            max_area = area;
        }
    }
    // Draw contour only for visualisation purposes
    //drawContours(img, contours[max_index], static_cast<int>(i), Scalar(0, 0, 255), 1, 8, hierarchy, 0);

    return contours[max_index];
}
