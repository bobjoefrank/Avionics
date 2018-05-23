#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "classifier.h"

cv::Mat CropOcrImage(std::vector<std::vector<cv::Point> > ocr_contours, int max_index, cv::Mat img, std::vector<cv::Vec4i> hierarchy, int letter_min_area, bool white_mask){

    //fill in max contour which will be the outermost outline of the letter
    cv::Mat ocr_image = cv::Mat::zeros(img.size(), CV_8UC3);
    if(white_mask){
        ocr_image.setTo(cv::Scalar::all(255));
    }
    cv::fillConvexPoly( ocr_image, ocr_contours[max_index], cv::Scalar(255, 255, 255));
    imshow( "just_the_outermost", ocr_image);
    char window_name[80];
    // for (auto vec : hierarchy){
    //     std::cout << vec << std::endl;
    // }

    //fill in the other contours which may or may not be the holes of the letter by alternation
    double area;
    bool alternate_color_inversion = true;
    cv::Scalar color;
    cv::RNG rng(12345);
    for (int i = ocr_contours.size()-1; i >= 0; i--){
        if(i != max_index && i < max_index){
            if(hierarchy[i][3] == -1){
                area = contourArea(ocr_contours[i]);
                if (area >= letter_min_area){
                    if (alternate_color_inversion){
                        color = cv::Scalar(0,0,0);
                        alternate_color_inversion = false;
                    } else {
                        color = cv::Scalar(255,255,255);
                    }
                    cv::fillConvexPoly( ocr_image, ocr_contours[i], color);
                    sprintf(window_name, "step_number:%d", (int)i);
                    imshow( window_name, ocr_image);
                }
            }
        }
    }

    //crop out bounded box of letter
    cv::Rect bounding_box = boundingRect(ocr_contours[max_index]);
    cv::Mat ocr_cropped = ocr_image(bounding_box);
    if(white_mask){
        bitwise_not(ocr_cropped,ocr_cropped);
    }

    return ocr_cropped;
}

cv::Mat ResizeOcrImage(cv::Mat ocr_image, bool rotated_45){
    //create new image with some space between edge of picture
    cv::Mat ocr_resized(ocr_image.size().height*1.3, ocr_image.size().width*1.3, CV_8UC3, cv::Scalar(0, 0, 0));
    //rotating by 45 degrees causes extra border so this will make the border smaller
    if(rotated_45){
        cv::resize(ocr_resized, ocr_resized, cv::Size(ocr_resized.rows/1.15,ocr_resized.cols/1.15));
    }

    //find center of new created image
    cv::Point ocr_center(ocr_resized.cols/2, ocr_resized.rows/2);

    //compute the rectangle centered in the image, same size as box
    cv::Rect center_box(ocr_center.x - ocr_image.cols/2, ocr_center.y - ocr_image.rows/2, ocr_image.cols, ocr_image.rows);

    //copy the letter into the resized image
    ocr_image.copyTo(ocr_resized(center_box));

    //resize image to 28x28
    cv::resize(ocr_resized, ocr_resized, cv::Size(28,28));

    return ocr_resized;
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