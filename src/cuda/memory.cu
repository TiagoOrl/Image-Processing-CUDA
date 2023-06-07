#include "memory.cuh"

void Memory::prepare_allocate1(
    unsigned char ** h_channelIn, 
    unsigned char ** d_channelIn,
    unsigned char ** d_channelOut,
    int img_size) {
        checkCudaErrors(cudaMalloc(d_channelIn, sizeof(unsigned char) * img_size));
        checkCudaErrors(cudaMalloc(d_channelOut, sizeof(unsigned char) * img_size));
        checkCudaErrors(cudaMemcpy( *d_channelIn, *h_channelIn, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemset(*d_channelOut, 0, sizeof(unsigned char) * img_size));
}

void Memory::prepare_allocate3(unsigned char ** h_channelR, unsigned char ** h_channelG, unsigned char ** h_channelB, 
    unsigned char ** d_channelR, unsigned char ** d_channelG, unsigned char ** d_channelB,
    unsigned char ** d_channelR_out, unsigned char ** d_channelG_out, unsigned char ** d_channelB_out,
    int img_size) {
        checkCudaErrors(cudaMalloc(d_channelR, sizeof(unsigned char) * img_size));
        checkCudaErrors(cudaMalloc(d_channelG, sizeof(unsigned char) * img_size));
        checkCudaErrors(cudaMalloc(d_channelB, sizeof(unsigned char) * img_size));

        checkCudaErrors(cudaMalloc(d_channelR_out, sizeof(unsigned char) * img_size));
        checkCudaErrors(cudaMalloc(d_channelG_out, sizeof(unsigned char) * img_size));
        checkCudaErrors(cudaMalloc(d_channelB_out, sizeof(unsigned char) * img_size));

        checkCudaErrors(cudaMemset(*d_channelR_out, 0, sizeof(unsigned char) * img_size));
        checkCudaErrors(cudaMemset(*d_channelG_out, 0, sizeof(unsigned char) * img_size));
        checkCudaErrors(cudaMemset(*d_channelB_out, 0, sizeof(unsigned char) * img_size));

        checkCudaErrors(cudaMemcpy( *d_channelR, *h_channelR, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy( *d_channelG, *h_channelG, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy( *d_channelB, *h_channelB, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice));
}