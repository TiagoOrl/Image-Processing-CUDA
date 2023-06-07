NVCC=nvcc
GCC=g++

OPENCV_INCLUDEPATH=/usr/include/opencv4
OPENCV_LIBS= -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

NVCC_OPTS= -O3 -g -G -Xcompiler -Wall -Xcompiler -Wextra -m64 

GCC_OPTS= -O3 -g -G -m64 



process: main.o kernel_processing.o memory.o image.o
	$(NVCC) -o process main.o kernel_processing.o memory.o image.o $(OPENCV_LIBS)

main.o: main.cpp src/utils/timer.h src/utils/utils.h 
	$(GCC) -c main.cpp -I $(OPENCV_INCLUDEPATH)

image.o: src/classes/image.hpp memory.o
	$(GCC) -c src/classes/image.cpp -I $(OPENCV_INCLUDEPATH)

memory.o: src/cuda/memory.cuh
	$(NVCC) -c src/cuda/memory.cu

kernel_processing.o: src/cuda/kernel_processing.cu src/utils/utils.h
	$(NVCC) -c src/cuda/kernel_processing.cu 

cl:
	rm *.o process *.jpg

clean:
	rm *.o process *.jpg




