
#include "../utils/timer.h"
#include "../utils/utils.h"
#include <stdio.h>


__global__
void k_grayScale (
    u_char * r_in, u_char * g_in, u_char * b_in,
    u_char * r_out, u_char * g_out, u_char * b_out,
    int height, int width) {
        
        int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
        int tIdy = threadIdx.y + blockIdx.y * blockDim.y;

        if (tIdx >= width || tIdy >= height) return;
        int i = width * tIdy + tIdx;

        int avg = (r_in[i] + g_in[i] + b_in[i]) / 3;

        r_out[i] = avg;
        g_out[i] = avg;
        b_out[i] = avg;
    }

__global__
void k_blur (
    u_char * r_in, u_char * g_in, u_char * b_in,
    u_char * r_out, u_char * g_out, u_char * b_out,
    int height, int width) {
    
    int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int tIdy = threadIdx.y + blockIdx.y * blockDim.y;
    int size = width * height;

    if (tIdx >= width || tIdy >= height) return;
    int index = width * tIdy + tIdx;


    float kernel[] = {
        1, 4,  6,  4,  1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1, 4,  6,  4,  1
    };

    const uint length = sizeof(kernel) / sizeof(kernel[0]);

    for (uint i = 0; i < length; i++)
    {
        kernel[i] = kernel[i] / 256;
    }

    int rowCount = -2;
    int colCount = -2;


    u_char rSum = 0;
    u_char gSum = 0;
    u_char bSum = 0;
    for (uint i = 0; i < length; i++)
    {
        if (i % 5 == 0 && i != 0)
            rowCount++;

        int kIndex = index + rowCount * width + colCount;
        if (kIndex < 0 || kIndex >= size)    
            continue;

        rSum += r_in[kIndex] * kernel[i];
        gSum += g_in[kIndex] * kernel[i];
        bSum += b_in[kIndex] * kernel[i];


        colCount++;
        if (colCount == 2)
            colCount = -2;

    }
    r_out[index] = rSum;
    g_out[index] = gSum;
    b_out[index] = bSum;
}

__global__
void k_sobel (
            u_char * r_in, u_char * g_in, u_char * b_in,
            u_char * r_out, u_char * g_out, u_char * b_out,
            int height, int width) {
    
    int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int tIdy = threadIdx.y + blockIdx.y * blockDim.y;
    size_t size = height * width;

    if (tIdx >= width || tIdy >= height) return;

    int index = width * tIdy + tIdx;

    float hMask[] = {
        1.0f, 0.0f, -1.0f,
        2.0f, 0.0f, -2.0f,
        1.0f, 0.0f, -1.0f
    };

    float vMask[] = {
        1.0f, 2.0f, 1.0f,
        0.0f, 0.0f, 0.0f,
        -1.0f, -2.0f, -1.0f
    };

    int hRSum = 0;
    int vRSum = 0;
    int hGSum = 0;
    int vGSum = 0;
    int hBSum = 0;
    int vBSum = 0;
    int rowCount = -1;
    int colCount = -1;

    for (size_t i = 0; i < 9; i++)
    {
        if (i % 3 == 0 && i != 0)
            rowCount++;
        
        int kIndex = index + rowCount * width + colCount;
        if (kIndex < 0 || kIndex >= size)
            continue;
        
        hRSum += r_in[kIndex] * hMask[i];
        vRSum += r_in[kIndex] * vMask[i];

        hGSum += g_in[kIndex] * hMask[i];
        vGSum += g_in[kIndex] * vMask[i];

        hBSum += b_in[kIndex] * hMask[i];
        vBSum += b_in[kIndex] * vMask[i];

        colCount++;
        if (colCount == 1)
            colCount = -1;
    }

    r_out[index] = (unsigned char) abs( (hRSum + vRSum) / 2);
    g_out[index] = (unsigned char) abs( (hGSum + vGSum) / 2);
    b_out[index] = (unsigned char) abs( (hBSum + vBSum) / 2);
}

__global__
void k_sobelBW (
            unsigned char * in_channel, 
            unsigned char * out_channel,
            int height, int width) {
    
    int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int tIdy = threadIdx.y + blockIdx.y * blockDim.y;
    size_t size = width * height;

    if (tIdx >= width || tIdy >= height) return;

    int index = width * tIdy + tIdx;

    float hMask[] = {
        1.0f, 0.0f, -1.0f,
        2.0f, 0.0f, -2.0f,
        1.0f, 0.0f, -1.0f
    };

    float vMask[] = {
        1.0f, 2.0f, 1.0f,
        0.0f, 0.0f, 0.0f,
        -1.0f, -2.0f, -1.0f
    };

    int hSum = 0;
    int vSum = 0;
    int rowCount = -1;
    int colCount = -1;

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

    for (size_t i = 0; i < 9; i++)
    {
        if (i % 3 == 0 && i != 0)
            rowCount++;
        
        int kIndex = index + rowCount * width + colCount;
        if (kIndex < 0 || kIndex >= size)
            continue;
        
        hSum += in_channel[kIndex] * hMask[i];
        vSum += in_channel[kIndex] * vMask[i];

        colCount++;
        if (colCount == 1)
            colCount = -1;
    }

    out_channel[index] = (unsigned char) abs( (hSum + vSum) / 2);
}



void cuda_blur(
    u_char * d_inR, u_char * d_inG, u_char * d_inB,
    u_char * d_outR, u_char * d_outG, u_char * d_outB,
    u_char * h_outR, u_char * h_outG, u_char * h_outB,
    int height, int width, 
    int blockWidth
)   {
    int img_size = width * height;
    int numBlocksX = width / blockWidth + 1;
    int numBlocksY = height / blockWidth + 1;
    const dim3 threadsPerBlock(blockWidth, blockWidth, 1);
    const dim3 numBlocks(numBlocksX, numBlocksY, 1);

    GpuTimer timer;

    timer.Start();

    k_blur<<<numBlocks, threadsPerBlock>>>(
        d_inR, d_inG, d_inB,
        d_outR, d_outG, d_outB, 
        height, width
    );
    cudaDeviceSynchronize();

    timer.Stop();
    printf("elapsed: %f ms\n", timer.Elapsed());

    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(h_outR, d_outR, sizeof(u_char) * img_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_outG, d_outG, sizeof(u_char) * img_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_outB, d_outB, sizeof(u_char) * img_size, cudaMemcpyDeviceToHost));

    cudaFree(d_inR);
    cudaFree(d_inG);
    cudaFree(d_inB);

    cudaFree(d_outR);
    cudaFree(d_outG);
    cudaFree(d_outB);
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

void cuda_grayScale(
    unsigned char * d_inR, unsigned char * d_inG, unsigned char * d_inB,
    unsigned char * d_outR, unsigned char * d_outG, unsigned char * d_outB,
    unsigned char * h_channelR_out, unsigned char * h_channelG_out, unsigned char * h_channelB_out,
    int height, int width, 
    int blockWidth
) {
    int img_size = width * height;
    int numBlocksX = width / blockWidth + 1;    
    int numBlocksY = height / blockWidth + 1;
    const dim3 threadsPerBlock (blockWidth, blockWidth, 1);
    const dim3 numBlocks (numBlocksX, numBlocksY, 1);
    GpuTimer timer;

    timer.Start();
    k_grayScale <<< numBlocks, threadsPerBlock >>> (
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
