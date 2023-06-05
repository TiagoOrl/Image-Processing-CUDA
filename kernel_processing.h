

void cuda_sobel( unsigned char * d_inR, unsigned char * d_inG, unsigned char * d_inB,
                unsigned char * d_outR, unsigned char * d_outG, unsigned char * d_outB,
                int height, int width);

void cuda_sobelBW( unsigned char * dIn, unsigned char * dOut, int height, int width);