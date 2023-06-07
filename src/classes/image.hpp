#ifndef IMAGE_CLASS_H
#define IMAGE_CLASS_H

#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core/types.hpp>

#include "../cuda/memory.cuh"
#include "../cuda/kernel_processing.h"

class Image
{
    public:
        static void sobelBW(cv::Mat &imgInput, cv::Mat &imgOutput);
        static void sobel(cv::Mat &imgInput, cv::Mat &imgOutput);
        static void output_image(const std::string output_file, cv::Mat out_image);
        static void saveImg(const std::string output_file, cv::Mat out_image);
};

#endif