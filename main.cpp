#include <string>

#include "src/utils/utils.h"
#include "src/classes/image.hpp"

int main(int argc, char const *argv[])
{
    if (argc < 4) {
        printf("Usage: process --filter inputfile outputname [-s]  \n");
        return EXIT_FAILURE;
    }
    
    std::string filter(argv[1]);
    std::string input_file(argv[2]);
    std::string outputName("");
    std::string saveFlag("");

    if (argc == 5) {
        outputName = argv[argc - 2];
        saveFlag = argv[argc - 1];
    }

    if (argc == 4) {
        outputName = argv[argc - 1];
    }

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

    else if (filter.compare("--blur") == 0) {
        Image::blur(imgInput, imgOutput);
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