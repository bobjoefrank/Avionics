//
//  find_pixel_color.cpp
//  object_detection_project_1
//
//  Created by Phillip Lai on 11/1/17.
//  Copyright © 2017 Phillip Lai. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
Mat image, src;
Mat HSV;
char window_name[20]="Pixel Value Demo";

static void onMouse( int event, int x, int y, int f, void* ){
    image=src.clone();
    Vec3b pix=HSV.at<Vec3b>(y,x);
    int H=pix.val[0];
    int S=pix.val[1];
    int V=pix.val[2];


    char name[30];
    sprintf(name,"H=%d",H);
    putText(image,name, Point(3640,130) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,255,0), 2,8,false );

    sprintf(name,"S=%d",S);
    putText(image,name, Point(3640,170) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,255,0), 2,8,false );

    sprintf(name,"V=%d",V);
    putText(image,name, Point(3640,210) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,255,0), 2,8,false );

    sprintf(name,"X=%d",x);
    putText(image,name, Point(3640,300) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,255,0), 2,8,false );

    sprintf(name,"Y=%d",y);
    putText(image,name, Point(3640,340) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,255,0), 2,8,false );
    imshow( window_name, image );
}



int main( int argc, char** argv )
{
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );

    src = imread(argv[1]);
    imshow( window_name, src );
    cvtColor(src,HSV,CV_BGR2HSV);
    setMouseCallback( window_name, onMouse, 0 );

    waitKey(0);

    return 0;
}
