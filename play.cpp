#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>

#include "videoprocessor.h"
#include "BGFGSegmentor.h"

//g++ -I/usr/local/opencv2/include/ -o foreground foreground.cpp  -L/usr/local/opencv2/lib -lopencv_objdetect -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_imgproc -lopencv_flann -lopencv_core -lopencv_nonfree

int main()
{
    // Open the video file
    //cv::VideoCapture capture("man1.avi");
    // check if video successfully opened
    //if (!capture.isOpened())
    //    return 0;

    // current video frame
    cv::Mat frame; 
    // foreground binary image
    cv::Mat foreground;
    // background image
    cv::Mat background;

    cv::namedWindow("Extracted Foreground");

    // The Mixture of Gaussian object
    // used with all default parameters
    cv::BackgroundSubtractorMOG mog;

    bool stop(false);
    // for all frames in video
    for(i=0;i<20,i++) {

        // read next frame if any
        //if (!capture.read(frame))
        //    break;

        // update the background
        // and return the foreground
        mog(frame,foreground,0.01);
        
        // Complement the image
        //cv::threshold(foreground,foreground,128,255,cv::THRESH_BINARY_INV);

        // show foreground and background
        cv::imshow("Extracted Foreground",foreground);

        // introduce a delay
        // or press key to stop
        if (cv::waitKey(10)>=0)
                stop= true;
    }

    cv::waitKey();

    // Create video procesor instance
    VideoProcessor processor;

    // Create background/foreground segmentor 
    BGFGSegmentor segmentor;
    segmentor.setThreshold(25);

    // Open video file
    processor.setInput("man1.avi");

    // set frame processor
    processor.setFrameProcessor(&segmentor);

    // Declare a window to display the video
    processor.displayOutput("Extracted Foreground");

    // Play the video at the original frame rate
    processor.setDelay(1000./processor.getFrameRate());

    // Start the process
    processor.run();

    cv::waitKey();
}