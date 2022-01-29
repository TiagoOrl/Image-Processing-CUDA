void output_image(const std::string &output_file, cv::Mat out_image);

void prepare_allocate1(uchar ** h_channelIn, 
                      uchar ** d_channelIn,
                      uchar ** d_channelOut,
                      int img_size);

void prepare_allocate3(uchar ** h_channelR, uchar ** h_channelG, uchar ** h_channelB, 
                      uchar ** d_channelR, uchar ** d_channelG, uchar ** d_channelB,
                      uchar ** d_channelR_out, uchar ** d_channelG_out, uchar ** d_channelB_out,
                      int img_size);

void sobel(cv::Mat &imgInput, cv::Mat &imgOutput);

void sobelBW(cv::Mat &imgInput, cv::Mat &imgOutput);