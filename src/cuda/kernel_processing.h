

void cuda_sobel( 
    unsigned char * d_inR, unsigned char * d_inG, unsigned char * d_inB,
    unsigned char * d_outR, unsigned char * d_outG, unsigned char * d_outB,
    unsigned char * h_channelR_out, unsigned char * h_channelG_out, unsigned char * h_channelB_out,
    int height, int width, 
    int blockWidth
);

void cuda_sobelBW( 
    unsigned char * dIn, 
    unsigned char * dOut, 
    int height, int width, 
    int blockwidth, 
    unsigned char * h_channelOut
);

void cuda_blur(
    u_char * d_inR, u_char * d_inG, u_char * d_inB,
    u_char * d_outR, u_char * d_outG, u_char * d_outB,
    u_char * h_outR, u_char * h_outG, u_char * h_outB,
    int height, int width, 
    int blockWidth
);