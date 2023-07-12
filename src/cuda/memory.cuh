#ifndef MEMORY_CLASS_H
#define MEMORY_CLASS_H

#include "../utils/utils.h"
typedef unsigned char uchar;

class Memory
{
    public:
        static void prepare_allocate1(uchar ** h_channelIn, 
                        uchar ** d_channelIn,
                        uchar ** d_channelOut,
                        int img_size);

        static void prepare_allocate3(uchar ** h_channelR, uchar ** h_channelG, uchar ** h_channelB, 
                            uchar ** d_channelR, uchar ** d_channelG, uchar ** d_channelB,
                            uchar ** d_channelR_out, uchar ** d_channelG_out, uchar ** d_channelB_out,
                            int img_size);
};

#endif