#!/bin/sh
python script.py 4 4 8
make #Compiling our src.c file
rm -f dataBcast.txt #Deleting in case it exists, which happens on reruns of the script.
rm -f dataReduce.txt #Deleting in case it exists, which happens on reruns of the script.
rm -f dataGather.txt #Deleting in case it exists, which happens on reruns of the script.

for execution in 1 2 3 4 5 6 7 8 9 10
do
	for P in 4 16
	do
		for ppn in 1 8
		do
			for D in 16 256 2048
			do
				num=$(($P*$ppn))
				echo "Running mpiexec -np $num -f hostfile ./code $D BCast"
				mpiexec -np $num -f hostfile ./code $D 0 >> dataBcast.txt
				echo "Running mpiexec -np $num -f hostfile ./code $D Reduce"
				mpiexec -np $num -f hostfile ./code $D 1 >> dataReduce.txt
				echo "Running mpiexec -np $num -f hostfile ./code $D Gather"
				mpiexec -np $num -f hostfile ./code $D 2 >> dataGather.txt		
			done
		done
	done	
done
echo "Simulation Results have been saved!"
echo "Generating Plots, The Script requires matplotlib, pandas, numpy, seaborn packages."
python3 plot.py dataBcast.txt plot_Bcast.png
python3 plot.py dataReduce.txt plot_Reduce.png
python3 plot.py dataGather.txt plot_Gather.png
echo "Plots files would have been generated if and only if seaborn package supports catplot."