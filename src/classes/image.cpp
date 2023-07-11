#include "image.hpp"

void Image::sobelBW(cv::Mat &imgInput, cv::Mat &imgOutput) {
    cv::cvtColor(imgInput, imgInput, cv::COLOR_RGBA2GRAY);
    int img_size = imgInput.rows * imgInput.cols;

    cv::Mat in_rgb[1];
    cv::Mat out_rgb[1];

    cv::split(imgInput, in_rgb);
    cv::split(imgInput, out_rgb);

    // h_ refers to data in host (cpu/RAM)
    // d_ refers to data in compute device (gpu/VRAM)

    uchar * h_channelIn = in_rgb[0].data;
    uchar * h_channelOut = out_rgb[0].data;

    uchar * d_channelIn;
    uchar * d_channelOut;

    Memory::prepare_allocate1(&h_channelIn, &d_channelIn, &d_channelOut, img_size);

    int width = imgInput.cols;
    int height = imgInput.rows;
    int blockwidth = 16;   

    cuda_sobelBW(d_channelIn, d_channelOut, height, width, blockwidth, h_channelOut);

    out_rgb[0].data = h_channelOut; 

    cv::merge(out_rgb, 1, imgOutput);
}

void Image::sobel(cv::Mat &imgInput, cv::Mat &imgOutput) {
    cv::cvtColor(imgInput, imgInput, cv::COLOR_RGBA2RGB);
    int img_size = imgInput.rows * imgInput.cols;

    cv::Mat in_rgb[3];
    cv::Mat out_rgb[3];

    cv::split(imgInput, in_rgb);
    cv::split(imgInput, out_rgb);

    // h_ refers to data in host (cpu/RAM)
    // d_ refers to data in compute device (gpu/VRAM)

    uchar * h_channelB_in = in_rgb[2].data;
    uchar * h_channelG_in = in_rgb[1].data;
    uchar * h_channelR_in = in_rgb[0].data;

    uchar * h_channelB_out = out_rgb[2].data;
    uchar * h_channelG_out = out_rgb[1].data;
    uchar * h_channelR_out = out_rgb[0].data;

    uchar * d_channelB_in;
    uchar * d_channelG_in;
    uchar * d_channelR_in;

    uchar * d_channelB_out;
    uchar * d_channelG_out;
    uchar * d_channelR_out;

    Memory::prepare_allocate3(&h_channelR_in, &h_channelG_in, &h_channelB_in, 
                     &d_channelR_in, &d_channelG_in, &d_channelB_in,
                     &d_channelR_out, &d_channelG_out, &d_channelB_out, 
                     img_size);

    
    int width = imgInput.cols;
    int height = imgInput.rows;
    int blockwidth = 16;   

    cuda_sobel( 
        d_channelR_in, d_channelG_in, d_channelB_in, 
        d_channelR_out, d_channelG_out, d_channelB_out, 
        h_channelR_out, h_channelG_out, h_channelB_out,
        imgInput.rows, imgInput.cols,
        blockwidth
    );               

    out_rgb[0].data = h_channelR_out; 
    out_rgb[1].data = h_channelG_out;
    out_rgb[2].data = h_channelB_out;

    cv::merge(out_rgb, 3, imgOutput);
}

void Image::blur(cv::Mat &imgInput, cv::Mat &imgOutput)
{
    cv::cvtColor(imgInput, imgInput, cv::COLOR_RGBA2RGB);
    int imgSize = imgInput.rows * imgInput.cols;

    cv::Mat in_rgb[3];
    cv::Mat out_rgb[3];

    cv::split(imgInput, in_rgb);
    cv::split(imgInput, out_rgb);

    uchar * h_channelB_in = in_rgb[2].data;
    uchar * h_channelG_in = in_rgb[1].data;
    uchar * h_channelR_in = in_rgb[0].data;

    uchar * h_channelB_out = out_rgb[2].data;
    uchar * h_channelG_out = out_rgb[1].data;
    uchar * h_channelR_out = out_rgb[0].data;

    uchar * d_channelB_in;
    uchar * d_channelG_in;
    uchar * d_channelR_in;

    uchar * d_channelB_out;
    uchar * d_channelG_out;
    uchar * d_channelR_out;

    Memory::prepare_allocate3(
            &h_channelR_in, &h_channelG_in, &h_channelB_in,
            &d_channelR_in, &d_channelG_in, & d_channelB_in,
            &d_channelR_out, &d_channelG_out, &d_channelB_out,
            imgSize
    );

    int width  = imgInput.cols;
    int height = imgInput.rows;
    int blockWidth = 16;

    cuda_blur(
        d_channelR_in, d_channelG_in, d_channelB_in,
        d_channelR_out, d_channelG_out, d_channelB_out,
        h_channelR_out, h_channelG_out, h_channelB_out,
        height, width, blockWidth
    );

    out_rgb[0].data = h_channelR_out;
    out_rgb[1].data = h_channelG_out;
    out_rgb[2].data = h_channelB_out;

    cv::merge(out_rgb, 3, imgOutput);
}

void Image::output_image(const std::string output_file, cv::Mat out_image) {

    cv::imshow(output_file, out_image);
    char k;

    while (k != 'q'){
        k = cv::waitKey(0); // img_input for a keystroke in the window
    }
}

void Image::saveImg(const std::string output_file, cv::Mat out_image) {
    // save the image
    cv::imwrite(output_file + ".jpg", out_image);
}