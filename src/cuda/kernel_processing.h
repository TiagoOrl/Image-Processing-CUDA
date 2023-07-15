

void cuda_sobel( 
    u_char * d_inR, u_char * d_inG, u_char * d_inB,
    u_char * d_outR, u_char * d_outG, u_char * d_outB,
    u_char * h_channelR_out, u_char * h_channelG_out, u_char * h_channelB_out,
    int height, int width, 
    int blockWidth
);

void cuda_sobelBW( 
    u_char * dIn, 
    u_char * dOut, 
    int height, int width, 
    int blockwidth, 
    u_char * h_channelOut
);

void cuda_blur(
    u_char * d_inR, u_char * d_inG, u_char * d_inB,
    u_char * d_outR, u_char * d_outG, u_char * d_outB,
    u_char * h_outR, u_char * h_outG, u_char * h_outB,
    int height, int width, 
    int blockWidth
);

void cuda_grayScale(
    u_char * d_inR, u_char * d_inG, u_char * d_inB,
    u_char * d_outR, u_char * d_outG, u_char * d_outB,
    u_char * h_channelR_out, u_char * h_channelG_out, u_char * h_channelB_out,
    int height, int width, 
    int blockWidth
);