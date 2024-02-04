CC = mpicc

FP_LIBS = -lm
GD_LIBS = -lm

FP_OBJS = MPI_Fempois.o
GD_OBJS = GridDist.o

all: MPI_Fempois GridDist

clean:
	rm -f *.o

# Define the default target
all:

# Define the target to build the executable
%: %.c
	mpicc -o $@ $<

# Clean up the executable files
clean:
	rm -f *



