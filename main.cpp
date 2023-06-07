
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/core.hpp>

#include <string>

#include "utils.h"
#include "image.hpp"

int main(int argc, char const *argv[])
{
    if (argc != 5) {
        printf("Usage: process --filter inputfile outputname [-s | -a]  \n");
        return EXIT_FAILURE;
    }
    
    std::string filter = argv[1];
    std::string input_file(argv[2]);
    std::string outputName(argv[argc - 2]);
    std::string saveFlag(argv[argc - 1]);
    cv::Mat imgInput = cv::imread(input_file.c_str(), cv::IMREAD_REDUCED_COLOR_2);
    cv::Mat imgOutput;


    cv::imshow("INPUT", imgInput);


    if (imgInput.empty() ) {
        std::cerr << "cound not open file: " << input_file << "\n";
        exit(1);
    }


    if (filter.compare("--sobel") == 0) {
        Image::sobel(imgInput, imgOutput);
    }

    else if (filter.compare("--sobelBW") == 0) {
        Image::sobelBW(imgInput, imgOutput);
    } 

    else {
        std::cerr << "Usage:\n ./process [--sobel, --sobelBW...] input.jpg outputName\n\n";
        exit(1);
    }

    if (saveFlag.compare("-s") == 0)
        Image::saveImg( outputName, imgOutput);
    Image::output_image(outputName, imgOutput);

    return 0;
}