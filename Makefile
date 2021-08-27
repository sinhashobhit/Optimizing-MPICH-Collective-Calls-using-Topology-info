CC=mpicc #The compiler we are using
TARGET=code #The name of the executable
CFLAGS = -O3 #Proving O3 max level of Optimization. 
#I wish I could have used Clang compiler to build the library.
all:
	$(CC) $(CFLAGS) -o $(TARGET) src.c -lm

clean:
	rm $(TARGET)
