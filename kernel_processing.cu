
#include "utils.h"
#include "timer.h"
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
            int height, int width) {
    
    int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int tIdy = threadIdx.y + blockIdx.y * blockDim.y;
    int size = height * width;

    if (tIdx >= width || tIdy >= height) return;

    int index = width * tIdy + tIdx;

    if (
        index - 1 < 0 ||
        index + 1 > size ||
        index - width < 0 ||
        index + width > size ||
        index - width - 1 < 0 ||
        index - width + 1 < 0 ||
        index + width - 1 > size ||
        index + width + 1 > size
    ) return;

    int v_Rkernel = (
        r_in[index] * 0.0 +
                
        r_in[index - 1] * 2.0 +     // west
        r_in[index + 1] * -2.0 +     // east
        
        r_in[index - width] * 0.0 +    // north
        r_in[index + width] * 0.0 +    // south
        
        r_in[index - width - 1] * 1.0 +    // northwest
        r_in[index - width + 1] * -1.0 +    // northeast
        r_in[index + width - 1] * 1.0 +    // southwest
        r_in[index + width + 1] * -1.0 ) ;   // southeast

    int h_Rkernel = (
        r_in[index] * 0.0 +
                
        r_in[index - 1] * 0.0 +     // west
        r_in[index + 1] * 0.0 +     // east
        
        r_in[index - width] * 2.0 +    // north
        r_in[index + width] * -2.0 +    // south
        
        r_in[index - width - 1] * 1.0 +    // northwest
        r_in[index - width + 1] * 1.0 +    // northeast
        r_in[index + width - 1] * -1.0 +    // southwest
        r_in[index + width + 1] * -1.0 ) ;   // southeast

    r_out[index] = (unsigned char) abs( (h_Rkernel + v_Rkernel) / 2);

    int v_Gkernel = (
        g_in[index] * 0.0 +
                
        g_in[index - 1] * 2.0 +     // west
        g_in[index + 1] * -2.0 +     // east
        
        g_in[index - width] * 0.0 +    // north
        g_in[index + width] * 0.0 +    // south
        
        g_in[index - width - 1] * 1.0 +    // northwest
        g_in[index - width + 1] * -1.0 +    // northeast
        g_in[index + width - 1] * 1.0 +    // southwest
        g_in[index + width + 1] * -1.0 ) ;   // southeast

    int h_Gkernel = (
        g_in[index] * 0.0 +
                
        g_in[index - 1] * 0.0 +     // west
        g_in[index + 1] * 0.0 +     // east
        
        g_in[index - width] * 2.0 +    // north
        g_in[index + width] * -2.0 +    // south
        
        g_in[index - width - 1] * 1.0 +    // northwest
        g_in[index - width + 1] * 1.0 +    // northeast
        g_in[index + width - 1] * -1.0 +    // southwest
        g_in[index + width + 1] * -1.0 ) ;   // southeast

    g_out[index] = (unsigned char) abs( (h_Gkernel + v_Gkernel) / 2);

    int v_Bkernel = (
        b_in[index] * 0.0 +
                
        b_in[index - 1] * 2.0 +     // west
        b_in[index + 1] * -2.0 +     // east
        
        b_in[index - width] * 0.0 +    // north
        b_in[index + width] * 0.0 +    // south
        
        b_in[index - width - 1] * 1.0 +    // northwest
        b_in[index - width + 1] * -1.0 +    // northeast
        b_in[index + width - 1] * 1.0 +    // southwest
        b_in[index + width + 1] * -1.0 ) ;   // southeast

    int h_Bkernel = (
        b_in[index] * 0.0 +
                
        b_in[index - 1] * 0.0 +     // west
        b_in[index + 1] * 0.0 +     // east
        
        b_in[index - width] * 2.0 +    // north
        b_in[index + width] * -2.0 +    // south
        
        b_in[index - width - 1] * 1.0 +    // northwest
        b_in[index - width + 1] * 1.0 +    // northeast
        b_in[index + width - 1] * -1.0 +    // southwest
        b_in[index + width + 1] * -1.0 ) ;   // southeast

    b_out[index] = (unsigned char) abs( (h_Bkernel + v_Bkernel) / 2);

}

__global__
void k_sobelBW (
            unsigned char * in_channel, 
            unsigned char * out_channel,
            int height, int width) {
    
    int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int tIdy = threadIdx.y + blockIdx.y * blockDim.y;
    int size = width * height;

    if (tIdx >= width || tIdy >= height) return;

    int index = width * tIdy + tIdx;

        if (
        index - 1 < 0 ||
        index + 1 > size ||
        index - width < 0 ||
        index + width > size ||
        index - width - 1 < 0 ||
        index - width + 1 < 0 ||
        index + width - 1 > size ||
        index + width + 1 > size
    ) return;

    int v_Rkernel = (
        in_channel[index] * 0.0 +
                
        in_channel[index - 1] * 2.0 +     // west
        in_channel[index + 1] * -2.0 +     // east
        
        in_channel[index - width] * 0.0 +    // north
        in_channel[index + width] * 0.0 +    // south
        
        in_channel[index - width - 1] * 1.0 +    // northwest
        in_channel[index - width + 1] * -1.0 +    // northeast
        in_channel[index + width - 1] * 1.0 +    // southwest
        in_channel[index + width + 1] * -1.0 ) ;   // southeast

    int h_Rkernel = (
        in_channel[index] * 0.0 +
                
        in_channel[index - 1] * 0.0 +     // west
        in_channel[index + 1] * 0.0 +     // east
        
        in_channel[index - width] * 2.0 +    // north
        in_channel[index + width] * -2.0 +    // south
        
        in_channel[index - width - 1] * 1.0 +    // northwest
        in_channel[index - width + 1] * 1.0 +    // northeast
        in_channel[index + width - 1] * -1.0 +    // southwest
        in_channel[index + width + 1] * -1.0 ) ;   // southeast

    out_channel[index] = (unsigned char) abs( (h_Rkernel + v_Rkernel) / 2);
}


void cuda_sobel( 
    unsigned char * d_inR, unsigned char * d_inG, unsigned char * d_inB,
    unsigned char * d_outR, unsigned char * d_outG, unsigned char * d_outB,
    unsigned char * h_channelR_out, unsigned char * h_channelG_out, unsigned char * h_channelB_out,
    int height, int width, 
    int blockWidth
)   {

    int img_size = width * height;
    int numBlocksX = width / blockWidth + 1;    
    int numBlocksY = height / blockWidth + 1;
    const dim3 threadsPerBlock (blockWidth, blockWidth, 1);
    const dim3 numBlocks (numBlocksX, numBlocksY, 1);
    GpuTimer timer;

    timer.Start();
    k_sobel <<< numBlocks, threadsPerBlock >>> (
        d_inR, d_inG, d_inB,
        d_outR, d_outG, d_outB,
        height, width);

    timer.Stop();
    printf("elapsed: %f ms\n", timer.Elapsed());
    
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors( cudaMemcpy( h_channelR_out, d_outR, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaMemcpy( h_channelG_out, d_outG, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaMemcpy( h_channelB_out, d_outB, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost) );


    cudaFree(d_inR);
    cudaFree(d_inG);
    cudaFree(d_inB);
    
    cudaFree(d_outR);
    cudaFree(d_outG);
    cudaFree(d_outB);
}

void cuda_sobelBW( 
    unsigned char * dIn, 
    unsigned char * dOut, 
    int height, int width, int blockwidth, 
    unsigned char * h_channelOut
)   {

    int numBlocksX = width / blockwidth + 1;    
    int numBlocksY = height / blockwidth + 1;
    const dim3 numBlocks (numBlocksX, numBlocksY, 1);
    const dim3 threadsPerBlock (blockwidth, blockwidth, 1);
    GpuTimer timer;
    

    std::cout << "num of blocks, x = " << numBlocks.x << " y = " << numBlocks.y << std::endl;

    timer.Start();
    k_sobelBW <<< numBlocks, threadsPerBlock >>> (
        dIn, 
        dOut,
        height, width
    );

    timer.Stop();
    printf("elapsed: %f ms\n", timer.Elapsed());
    
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors( cudaMemcpy( h_channelOut, dOut, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost) );

    cudaFree(dIn);
    cudaFree(dOut);
}
