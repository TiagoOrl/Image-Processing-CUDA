NVCC=nvcc
GCC=g++

OPENCV_INCLUDEPATH=/usr/include/opencv4
OPENCV_LIBS= -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

NVCC_OPTS= -O3 -g -G -Xcompiler -Wall -Xcompiler -Wextra -m64 

GCC_OPTS= -O3 -g -G -m64 



process: main.o kernel_processing.o memory.o image.o
	$(NVCC) -o process main.o kernel_processing.o memory.o image.o $(OPENCV_LIBS)

main.o: main.cpp timer.h utils.h 
	$(GCC) -c main.cpp -I $(OPENCV_INCLUDEPATH)

image.o: image.hpp memory.o
	$(GCC) -c image.cpp -I $(OPENCV_INCLUDEPATH)

memory.o: memory.cuh
	$(NVCC) -c memory.cu

kernel_processing.o: kernel_processing.cu utils.h
	$(NVCC) -c kernel_processing.cu 

cl:
	rm *.o process *.jpg

clean:
	rm *.o process *.jpg




