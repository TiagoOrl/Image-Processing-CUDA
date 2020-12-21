
#include "utils.h"
#include <stdio.h>


__global__
void k_blurChannel (unsigned char * input_channel, unsigned char * outputchannel, int numRows, int numCols) {
    
    int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int tIdy = threadIdx.y + blockIdx.y * blockDim.y;

    int index = numCols * tIdx + tIdy;

    if ( index >= numRows * numCols ) return;

    unsigned char new_color = (
        input_channel[index] * 0.5 +                    // itself

        input_channel[index - 1] * 0.125 +              // west
        input_channel[index + 1] * 0.125 +              // east

        input_channel[index - numCols] * 0.125 +        // north
        input_channel[index + numCols] * 0.125 +        // south

        input_channel[index - numCols - 1] * 0.0 +    // northwest
        input_channel[index - numCols + 1] * 0.0 +    // northeast

        input_channel[index + numCols - 1] * 0.0 +    // southwest
        input_channel[index + numCols + 1] * 0.0    // southeast

    );

    outputchannel[index] = new_color;
}

__global__
void k_sobel (unsigned char * input_channel, unsigned char * outputchannel, int numRows, int numCols) {
    
    int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int tIdy = threadIdx.y + blockIdx.y * blockDim.y;

    int index = numCols * tIdx + tIdy;

    if ( index >= numRows * numCols ) return;

    int v_kernel = (
        input_channel[index] * 0.0 +
                
        input_channel[index - 1] * 2.0 +     // west
        input_channel[index + 1] * -2.0 +     // east
        
        input_channel[index - numCols] * 0.0 +    // north
        input_channel[index + numCols] * 0.0 +    // south
        
        input_channel[index - numCols - 1] * 1.0 +    // northwest
        input_channel[index - numCols + 1] * -1.0 +    // northeast
        input_channel[index + numCols - 1] * 1.0 +    // southwest
        input_channel[index + numCols + 1] * -1.0 ) ;   // southeast

    int h_kernel = (
        input_channel[index] * 0.0 +
                
        input_channel[index - 1] * 0.0 +     // west
        input_channel[index + 1] * 0.0 +     // east
        
        input_channel[index - numCols] * 2.0 +    // north
        input_channel[index + numCols] * -2.0 +    // south
        
        input_channel[index - numCols - 1] * 1.0 +    // northwest
        input_channel[index - numCols + 1] * 1.0 +    // northeast
        input_channel[index + numCols - 1] * -1.0 +    // southwest
        input_channel[index + numCols + 1] * -1.0 ) ;   // southeast

       outputchannel[index] = (unsigned char) abs( (h_kernel + v_kernel) / 2);

}

__global__
void k_recombine_channels(unsigned char * red_out,
                        unsigned char * gree_out,
                        unsigned char * blue_out,
                        int numRows, int numCols,
                        uchar4 * outputImage ) {
    
    int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int tIdy = threadIdx.y + blockIdx.y * blockDim.y;

    const int index = tIdx + numCols * tIdy;

    if ( tIdx >= numCols || tIdy >= numRows) return;

    outputImage[index] = make_uchar4( red_out[index], gree_out[index], blue_out[index], 255 );

}

void cuda_blur( unsigned char * d_inR, unsigned char * d_inG, unsigned char * d_inB,
                unsigned char * d_outR, unsigned char * d_outG, unsigned char * d_outB,
                int rows, int cols) {

    
    // wrong blockwidth will cause illegal memory access

    int blockwidth = 16;   
    int blocksX = rows / blockwidth + 1;    
    int blocksY = cols / blockwidth + 1;
    const dim3 blockSize (blockwidth, blockwidth, 1);
    const dim3 gridSize (blocksX, blocksY, 1);
    

    k_sobel <<< gridSize, blockSize >>> (d_inR, d_outR, rows, cols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    k_sobel <<< gridSize, blockSize >>> (d_inG, d_outG, rows, cols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    k_sobel <<< gridSize, blockSize >>> (d_inB, d_outB, rows, cols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // k_recombine_channels <<< gridSize, blockSize >>> (d_outR, d_outG, d_outB, rows, cols, d_outputImage);
    // cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    cudaFree(d_inR);
    cudaFree(d_inG);
    cudaFree(d_inB);
}
