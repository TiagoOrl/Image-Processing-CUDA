#include "memory.cuh"

void Memory::prepare_allocate1(
    uchar ** h_channelIn, 
    uchar ** d_channelIn,
    uchar ** d_channelOut,
    int img_size) {
        checkCudaErrors(cudaMalloc(d_channelIn, sizeof(uchar) * img_size));
        checkCudaErrors(cudaMalloc(d_channelOut, sizeof(uchar) * img_size));
        checkCudaErrors(cudaMemcpy( *d_channelIn, *h_channelIn, sizeof(uchar) * img_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemset(*d_channelOut, 0, sizeof(uchar) * img_size));
}

void Memory::prepare_allocate3(
    uchar ** h_channelR, uchar ** h_channelG, uchar ** h_channelB, 
    uchar ** d_channelR, uchar ** d_channelG, uchar ** d_channelB,
    uchar ** d_channelR_out, uchar ** d_channelG_out, uchar ** d_channelB_out,
    int img_size) {
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