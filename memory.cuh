#ifndef MEMORY_CLASS_H
#define MEMORY_CLASS_H

#include "utils.h"

class Memory
{
    public:
        static void prepare_allocate1(unsigned char ** h_channelIn, 
                        unsigned char ** d_channelIn,
                        unsigned char ** d_channelOut,
                        int img_size);

        static void prepare_allocate3(unsigned char ** h_channelR, unsigned char ** h_channelG, unsigned char ** h_channelB, 
                            unsigned char ** d_channelR, unsigned char ** d_channelG, unsigned char ** d_channelB,
                            unsigned char ** d_channelR_out, unsigned char ** d_channelG_out, unsigned char ** d_channelB_out,
                            int img_size);
};

#endif