
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
void k_sobel (
            unsigned char * r_in, unsigned char * g_in, unsigned char * b_in,
            unsigned char * r_out, unsigned char * g_out, unsigned char * b_out,
            int numRows, int numCols) {
    
    int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int tIdy = threadIdx.y + blockIdx.y * blockDim.y;

    int index = numCols * tIdx + tIdy;

    if ( index >= numRows * numCols ) return;

    int v_Rkernel = (
        r_in[index] * 0.0 +
                
        r_in[index - 1] * 2.0 +     // west
        r_in[index + 1] * -2.0 +     // east
        
        r_in[index - numCols] * 0.0 +    // north
        r_in[index + numCols] * 0.0 +    // south
        
        r_in[index - numCols - 1] * 1.0 +    // northwest
        r_in[index - numCols + 1] * -1.0 +    // northeast
        r_in[index + numCols - 1] * 1.0 +    // southwest
        r_in[index + numCols + 1] * -1.0 ) ;   // southeast

    int h_Rkernel = (
        r_in[index] * 0.0 +
                
        r_in[index - 1] * 0.0 +     // west
        r_in[index + 1] * 0.0 +     // east
        
        r_in[index - numCols] * 2.0 +    // north
        r_in[index + numCols] * -2.0 +    // south
        
        r_in[index - numCols - 1] * 1.0 +    // northwest
        r_in[index - numCols + 1] * 1.0 +    // northeast
        r_in[index + numCols - 1] * -1.0 +    // southwest
        r_in[index + numCols + 1] * -1.0 ) ;   // southeast

    r_out[index] = (unsigned char) abs( (h_Rkernel + v_Rkernel) / 2);

    int v_Gkernel = (
        g_in[index] * 0.0 +
                
        g_in[index - 1] * 2.0 +     // west
        g_in[index + 1] * -2.0 +     // east
        
        g_in[index - numCols] * 0.0 +    // north
        g_in[index + numCols] * 0.0 +    // south
        
        g_in[index - numCols - 1] * 1.0 +    // northwest
        g_in[index - numCols + 1] * -1.0 +    // northeast
        g_in[index + numCols - 1] * 1.0 +    // southwest
        g_in[index + numCols + 1] * -1.0 ) ;   // southeast

    int h_Gkernel = (
        g_in[index] * 0.0 +
                
        g_in[index - 1] * 0.0 +     // west
        g_in[index + 1] * 0.0 +     // east
        
        g_in[index - numCols] * 2.0 +    // north
        g_in[index + numCols] * -2.0 +    // south
        
        g_in[index - numCols - 1] * 1.0 +    // northwest
        g_in[index - numCols + 1] * 1.0 +    // northeast
        g_in[index + numCols - 1] * -1.0 +    // southwest
        g_in[index + numCols + 1] * -1.0 ) ;   // southeast

    g_out[index] = (unsigned char) abs( (h_Gkernel + v_Gkernel) / 2);

    int v_Bkernel = (
        b_in[index] * 0.0 +
                
        b_in[index - 1] * 2.0 +     // west
        b_in[index + 1] * -2.0 +     // east
        
        b_in[index - numCols] * 0.0 +    // north
        b_in[index + numCols] * 0.0 +    // south
        
        b_in[index - numCols - 1] * 1.0 +    // northwest
        b_in[index - numCols + 1] * -1.0 +    // northeast
        b_in[index + numCols - 1] * 1.0 +    // southwest
        b_in[index + numCols + 1] * -1.0 ) ;   // southeast

    int h_Bkernel = (
        b_in[index] * 0.0 +
                
        b_in[index - 1] * 0.0 +     // west
        b_in[index + 1] * 0.0 +     // east
        
        b_in[index - numCols] * 2.0 +    // north
        b_in[index + numCols] * -2.0 +    // south
        
        b_in[index - numCols - 1] * 1.0 +    // northwest
        b_in[index - numCols + 1] * 1.0 +    // northeast
        b_in[index + numCols - 1] * -1.0 +    // southwest
        b_in[index + numCols + 1] * -1.0 ) ;   // southeast

    b_out[index] = (unsigned char) abs( (h_Bkernel + v_Bkernel) / 2);

}

__global__
void k_sobelBW (
            unsigned char * in_channel, 
            unsigned char * out_channel,
            int numRows, int numCols) {
    
    int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int tIdy = threadIdx.y + blockIdx.y * blockDim.y;

    int index = numCols * tIdx + tIdy;

    if ( index >= numRows * numCols ) return;

    int v_Rkernel = (
        in_channel[index] * 0.0 +
                
        in_channel[index - 1] * 2.0 +     // west
        in_channel[index + 1] * -2.0 +     // east
        
        in_channel[index - numCols] * 0.0 +    // north
        in_channel[index + numCols] * 0.0 +    // south
        
        in_channel[index - numCols - 1] * 1.0 +    // northwest
        in_channel[index - numCols + 1] * -1.0 +    // northeast
        in_channel[index + numCols - 1] * 1.0 +    // southwest
        in_channel[index + numCols + 1] * -1.0 ) ;   // southeast

    int h_Rkernel = (
        in_channel[index] * 0.0 +
                
        in_channel[index - 1] * 0.0 +     // west
        in_channel[index + 1] * 0.0 +     // east
        
        in_channel[index - numCols] * 2.0 +    // north
        in_channel[index + numCols] * -2.0 +    // south
        
        in_channel[index - numCols - 1] * 1.0 +    // northwest
        in_channel[index - numCols + 1] * 1.0 +    // northeast
        in_channel[index + numCols - 1] * -1.0 +    // southwest
        in_channel[index + numCols + 1] * -1.0 ) ;   // southeast

    out_channel[index] = (unsigned char) abs( (h_Rkernel + v_Rkernel) / 2);
}


void cuda_sobel( unsigned char * d_inR, unsigned char * d_inG, unsigned char * d_inB,
                unsigned char * d_outR, unsigned char * d_outG, unsigned char * d_outB,
                int rows, int cols) {

    
    // wrong blockwidth will cause illegal memory access

    int blockwidth = 16;   
    int numBlocksX = rows / blockwidth + 1;    
    int numBlocksY = cols / blockwidth + 1;
    const dim3 threadsPerBlock (blockwidth, blockwidth, 1);
    const dim3 totalBlocks (numBlocksX, numBlocksY, 1);
    

    k_sobel <<< totalBlocks, threadsPerBlock >>> (
        d_inR, d_inG, d_inB,
        d_outR, d_outG, d_outB,
        rows, cols);
    
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());
}

void cuda_sobelBW( 
                unsigned char * dIn, 
                unsigned char * dOut, 
                int rows, int cols) {

    // wrong blockwidth will cause illegal memory access

    int blockwidth = 16;   
    int numBlocksX = rows / blockwidth + 1;    
    int numBlocksY = cols / blockwidth + 1;
    const dim3 totalBlocks (numBlocksX, numBlocksY, 1);
    const dim3 threadsPerBlock (blockwidth, blockwidth, 1);
    

    k_sobelBW <<< totalBlocks, threadsPerBlock >>> (
        dIn, 
        dOut,
        rows, cols);
    
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());
}
