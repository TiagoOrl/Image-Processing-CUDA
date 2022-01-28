#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include "timer.h"
#include "utils.h"
#include <string>


void output_image(const std::string &output_file, cv::Mat out_image);

void prepare_allocate1(uchar ** h_channelIn, 
                      uchar ** d_channelIn,
                      uchar ** d_channelOut,
                      int img_size);

void prepare_allocate3(uchar ** h_channelR, uchar ** h_channelG, uchar ** h_channelB, 
                      uchar ** d_channelR, uchar ** d_channelG, uchar ** d_channelB,
                      uchar ** d_channelR_out, uchar ** d_channelG_out, uchar ** d_channelB_out,
                      int img_size);

void cuda_sobel( unsigned char * d_inR, unsigned char * d_inG, unsigned char * d_inB,
                unsigned char * d_outR, unsigned char * d_outG, unsigned char * d_outB,
                int rows, int cols);

void sobel(cv::Mat &imgInput, cv::Mat &imgOutput);

void cuda_sobelBW( unsigned char * dIn, unsigned char * dOut, int rows, int cols);

void sobelBW(cv::Mat &imgInput, cv::Mat &imgOutput);


int main(int argc, char const *argv[])
{
    if (argc != 4) {
        printf("Usage: process [--filter] inputfile outputname \n");
        return EXIT_FAILURE;
    }
    std::string filter = argv[1];
    std::string input_file(argv[2]);
    std::string outputName(argv[argc - 1]);
    cv::Mat imgInput = cv::imread(input_file.c_str(), cv::IMREAD_REDUCED_COLOR_2);
    cv::Mat imgOutput;


    cv::imshow("INPUT", imgInput);


    if (imgInput.empty() ) {
        std::cerr << "cound not open file: " << input_file << std::endl;
        exit(1);
    }


    if (filter.compare("--sobel") == 0) {
        sobel(imgInput, imgOutput);
    }

    else if (filter.compare("--sobelBW") == 0) {
        sobelBW(imgInput, imgOutput);
    } 

    else {
        std::cerr << "Usage:\n ./process [--sobel, --sobelBW...] input.jpg outputName" << std::endl;
        exit(1);
    }

    output_image(outputName, imgOutput);

    return 0;
}


void sobel(cv::Mat &imgInput, cv::Mat &imgOutput) {
    
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

    uchar * h_channelB_out = in_rgb[2].data;
    uchar * h_channelG_out = in_rgb[1].data;
    uchar * h_channelR_out = in_rgb[0].data;

    uchar * d_channelB_in;
    uchar * d_channelG_in;
    uchar * d_channelR_in;

    uchar * d_channelB_out;
    uchar * d_channelG_out;
    uchar * d_channelR_out;

    prepare_allocate3(&h_channelR_in, &h_channelG_in, &h_channelB_in, 
                     &d_channelR_in, &d_channelG_in, &d_channelB_in,
                     &d_channelR_out, &d_channelG_out, &d_channelB_out, 
                     img_size);

    
    GpuTimer timer;

    timer.Start();
    cuda_sobel( d_channelR_in, d_channelG_in, d_channelB_in, 
               d_channelR_out, d_channelG_out, d_channelB_out, 
               imgInput.rows, imgInput.cols);               
    timer.Stop();

    printf("elapsed: %f ms\n", timer.Elapsed());


    checkCudaErrors( cudaMemcpy( h_channelR_out, d_channelR_out, sizeof(uchar) * img_size, cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaMemcpy( h_channelG_out, d_channelG_out, sizeof(uchar) * img_size, cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaMemcpy( h_channelB_out, d_channelB_out, sizeof(uchar) * img_size, cudaMemcpyDeviceToHost) );

    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());

    cudaFree(d_channelB_in);
    cudaFree(d_channelG_in);
    cudaFree(d_channelR_in);
    
    cudaFree(d_channelR_out);
    cudaFree(d_channelG_out);
    cudaFree(d_channelB_out);

    out_rgb[0].data = h_channelR_out; 
    out_rgb[1].data = h_channelG_out;
    out_rgb[2].data = h_channelB_out;

    cv::merge(out_rgb, 3, imgOutput);
}


void sobelBW(cv::Mat &imgInput, cv::Mat &imgOutput) {
    
    cv::cvtColor(imgInput, imgInput, cv::COLOR_RGBA2GRAY);
    int img_size = imgInput.rows * imgInput.cols;

    cv::Mat in_rgb[1];
    cv::Mat out_rgb[1];

    cv::split(imgInput, in_rgb);
    cv::split(imgInput, out_rgb);

    // h_ refers to data in host (cpu/RAM)
    // d_ refers to data in compute device (gpu/VRAM)


    uchar * h_channelIn = in_rgb[0].data;
    uchar * h_channelOut = in_rgb[0].data;

    uchar * d_channelIn;
    uchar * d_channelOut;

    prepare_allocate1(&h_channelIn, &d_channelIn, &d_channelOut, img_size);

    GpuTimer timer;

    timer.Start();
    cuda_sobelBW( d_channelIn, d_channelOut, imgInput.rows, imgInput.cols);               
    timer.Stop();

    printf("elapsed: %f ms\n", timer.Elapsed());

    checkCudaErrors( cudaMemcpy( h_channelOut, d_channelOut, sizeof(uchar) * img_size, cudaMemcpyDeviceToHost) );

    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());


    cudaFree(d_channelIn);
    cudaFree(d_channelOut);


    out_rgb[0].data = h_channelOut; 

    cv::merge(out_rgb, 1, imgOutput);
}


void prepare_allocate3(uchar ** h_channelR, uchar ** h_channelG, uchar ** h_channelB, 
                      uchar ** d_channelR, uchar ** d_channelG, uchar ** d_channelB,
                      uchar ** d_channelR_out, uchar ** d_channelG_out, uchar ** d_channelB_out,
                      int img_size)
{

    checkCudaErrors(cudaMalloc(d_channelR, sizeof(uchar) * img_size));
    checkCudaErrors(cudaMalloc(d_channelG, sizeof(uchar) * img_size));
    checkCudaErrors(cudaMalloc(d_channelB, sizeof(uchar) * img_size));

    checkCudaErrors(cudaMalloc(d_channelR_out, sizeof(uchar) * img_size));
    checkCudaErrors(cudaMalloc(d_channelG_out, sizeof(uchar) * img_size));
    checkCudaErrors(cudaMalloc(d_channelB_out, sizeof(uchar) * img_size));

    checkCudaErrors(cudaMemset(*d_channelR_out, 0, sizeof(uchar) * img_size));
    checkCudaErrors(cudaMemset(*d_channelG_out, 0, sizeof(uchar) * img_size));
    checkCudaErrors(cudaMemset(*d_channelB_out, 0, sizeof(uchar) * img_size));

    checkCudaErrors(cudaMemcpy( *d_channelR, *h_channelR, sizeof(uchar) * img_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( *d_channelG, *h_channelG, sizeof(uchar) * img_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( *d_channelB, *h_channelB, sizeof(uchar) * img_size, cudaMemcpyHostToDevice));

}


void prepare_allocate1(uchar ** h_channelIn, 
                      uchar ** d_channelIn,
                      uchar ** d_channelOut,
                      int img_size)
{
    checkCudaErrors(cudaMalloc(d_channelIn, sizeof(uchar) * img_size));

    checkCudaErrors(cudaMalloc(d_channelOut, sizeof(uchar) * img_size));

    checkCudaErrors(cudaMemcpy( *d_channelIn, *h_channelIn, sizeof(uchar) * img_size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(*d_channelOut, 0, sizeof(uchar) * img_size));
}

void output_image(const std::string &output_file, cv::Mat out_image) {

    cv::imshow("OUTPUT", out_image);
    char k;

    while (k != 'q'){
        k = cv::waitKey(0); // img_input for a keystroke in the window
    }
    
    // output the image
    cv::imwrite(output_file + ".jpg", out_image);
}
