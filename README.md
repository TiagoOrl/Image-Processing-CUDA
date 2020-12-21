[english]

This is a program to scan images and process them in parallel in CUDA GPU's. The idea is that, we read an input image in 3 channel
 mode (RGB) using openCV, allocate memory in GPU for each channel and 3 additional in GPU for the output channel, then in the parallel region
(kernel call) we are able to read each array index of the input channels based on the thread/block index and create a new pixel based on
 then (then write into the output channel), after we processed each 3 channel and written into the 3 output channels, we recombine the 3
 output channels into an array of uchar4 (end of parallel region), we are back into the serial code and we use openCV again to 
show and write our output data. In the .cu file, I've implemented the Sobel and Gauss blur filter in parallel.

To generate the executable in Linux, just execute make inside terminal and:

$ ./process in-imgs/exameple.img outputimg.jpg

I've developed this program based of the great course CS344: Intro to Parallel Programming from NVIDIA:
https://www.youtube.com/playlist?list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2

Great video about Image Processing:
https://www.youtube.com/watch?v=mRM5Js3VLCk&t=1210s
