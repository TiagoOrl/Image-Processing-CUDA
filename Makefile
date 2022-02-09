NVCC=nvcc
GCC=g++

OPENCV_INCLUDEPATH=/usr/include/opencv4
OPENCV_LIBS= -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

NVCC_OPTS= -O3 -g -Xcompiler -Wall -Xcompiler -Wextra -m64 

GCC_OPTS= -O3 -g -m64 



process: main.o kernel_processing.o
	$(NVCC) -o process main.o kernel_processing.o $(OPENCV_LIBS) $(NVCC_OPTS)

main.o: main.cpp timer.h utils.h 
	$(GCC) -c main.cpp -I $(OPENCV_INCLUDEPATH)


kernel_processing.o: kernel_processing.cu utils.h
	$(NVCC) -c kernel_processing.cu 

clean:
	rm *.o process *.jpg




