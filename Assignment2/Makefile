all:
	echo "Don't forget to: module load xl_r spectrum-mpi cuda/11.2"
	xlc_r -Wall -O3 cla-serial.c -o cla-serial
	nvcc -O3 cla-parallel.cu -o cla-parallel
