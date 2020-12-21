

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include "timer.h"
#include "utils.h"
#include <string>



void cuda_process( unsigned char * d_inR, unsigned char * d_inG, unsigned char * d_inB,
                unsigned char * d_outR, unsigned char * d_outG, unsigned char * d_outB,
                int rows, int cols);


void prepare_allocate(uchar ** h_channelR, uchar ** h_channelG, uchar ** h_channelB, 
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

void output_image(const std::string &output_file, cv::Mat out_image) {

    cv::imshow("OUTPUT", out_image);
    char k;

    while (k != 'q'){
        k = cv::waitKey(0); // img_input for a keystroke in the window
    }
    
    // output the image
    cv::imwrite(output_file.c_str(), out_image);
}

int main(int argc, char const *argv[])
{
    if (argc != 3) {
        printf("Usage: process inputfile outputname.jpg \n");
        return EXIT_FAILURE;
    }

    std::string input_file(argv[1]);
    std::string output_file(argv[2]);
    cv::Mat cv_imageInput = cv::imread(input_file.c_str(), cv::IMREAD_REDUCED_COLOR_4);
    cv::Mat cv_img_output;


    cv::imshow("INPUT", cv_imageInput);


    if (cv_imageInput.empty() ) {
        std::cerr << "cound not open file: " << input_file << std::endl;
        exit(1);
    }

    // convert RGB to RGBA
    cv::cvtColor(cv_imageInput, cv_imageInput, cv::COLOR_RGBA2RGB);

    int img_size = cv_imageInput.rows * cv_imageInput.cols;

    cv::Mat in_rgb[3];
    cv::Mat out_rgb[3];

    cv::split(cv_imageInput, in_rgb);
    cv::split(cv_imageInput, out_rgb);

    // h_ refers to data/variables in host (cpu/RAM)
    // d_ refers to data/variables in compute device (gpu/VRAM)

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

    prepare_allocate(&h_channelR_in, &h_channelG_in, &h_channelB_in, 
                     &d_channelR_in, &d_channelG_in, &d_channelB_in,
                     &d_channelR_out, &d_channelG_out, &d_channelB_out, 
                     img_size);

    
    GpuTimer timer;

    timer.Start();
    cuda_process( d_channelR_in, d_channelG_in, d_channelB_in, 
               d_channelR_out, d_channelG_out, d_channelB_out, 
               cv_imageInput.rows, cv_imageInput.cols);               
    timer.Stop();

    printf("elapsed: %f ms\n", timer.Elapsed());


    checkCudaErrors( cudaMemcpy( h_channelR_out, d_channelR_out, sizeof(uchar) * img_size, cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaMemcpy( h_channelG_out, d_channelG_out, sizeof(uchar) * img_size, cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaMemcpy( h_channelB_out, d_channelB_out, sizeof(uchar) * img_size, cudaMemcpyDeviceToHost) );

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    cudaFree(d_channelB_in);
    cudaFree(d_channelG_in);
    cudaFree(d_channelR_in);
    
    cudaFree(d_channelR_out);
    cudaFree(d_channelG_out);
    cudaFree(d_channelB_out);

    out_rgb[0].data = h_channelR_out; 
    out_rgb[1].data = h_channelG_out;
    out_rgb[2].data = h_channelB_out;

    cv::merge(out_rgb, 3, cv_img_output);

    output_image(output_file, cv_img_output);

    return 0;
}
