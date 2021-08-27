/*
Source Code of Optimization of MPICH Collective Calls using Network Topology Info
Authors : Shobhit Sinha
Last Modified on 30th March, 11:33 A.M.
*/

	
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mpi.h"

/*
The Cluster Info Matrix stores the Network Topology. The nodes present in the same row have a hop distance of 2 and in different rows have a hop distance of 4. The Matrix has been padded with -1 if no of machines in a particular sub-cluster is less than 17
*/
int clusterInfo[6][17] = {
	{79,80,81,82,83,84,85,86,87,88,89,90,91,92,-1,-1,-1},
	{62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78},
	{45,47,48,49,50,51,52,53,54,56,58,59,60,61,-1,-1,-1},
	{33,34,35,36,37,38,39,40,41,42,43,44,46,-1,-1,-1,-1},
	{13,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,-1},
	{2,3,4,5,6,7,8,9,10,11,12,14,15,16,31,-1,-1}
};

/*Auxilliary or Helper Functions*/
void initialize_randomly(int N, double *matrix, int rank);
void display_array(int N, double *arr);


/*Function Prototypes for Optimized Functions*/

int MPI_Gather_optimized(double *sendbuf, int sendcount, MPI_Datatype sendtype, double *recvbuf, int recvcount,MPI_Datatype recvtype, int root, int param, MPI_Comm global_communicator,MPI_Comm sameHost_comm, MPI_Comm subCluster_comm, MPI_Comm final_comm);

int MPI_Bcast_optimized(void *buffer,int count,MPI_Datatype datatype,int root, MPI_Comm global_communicator,MPI_Comm sameHost_comm, MPI_Comm subCluster_comm, MPI_Comm final_comm);

int MPI_Reduce_optimized(void* send_buffer, void* receive_buffer, int count, MPI_Datatype datatype, MPI_Op operation,int root, MPI_Comm global_communicator,MPI_Comm sameHost_comm, MPI_Comm subCluster_comm, MPI_Comm final_comm);
int MPI_Alltoallv_optimized(double *sendbuf, int sendcount, MPI_Datatype sendtype, double *recvbuf, int recvcount,MPI_Datatype recvtype, int root, int param, MPI_Comm global_communicator,MPI_Comm sameHost_comm, MPI_Comm subCluster_comm, MPI_Comm final_comm){
	/*
	This is not a correct implementation of optimized Alltoallv().
	This provides a baseline comparison of available Alltoall and Topologically aware Alltoall. While implementing Alltoallv(), there was a lot of book-keeping involved in C language. The coding of Alltoallv_optimized() would have been super easy if I had C++ STL at my disposal.
	*/
	int size;
	MPI_Comm_size(global_communicator, &size);
	MPI_Gather_optimized(sendbuf, sendcount*size, sendtype, recvbuf, recvcount, recvtype, root, param ,global_communicator , sameHost_comm, subCluster_comm, final_comm);
	MPI_Bcast_optimized(recvbuf, sendcount*size, sendtype, root ,global_communicator , sameHost_comm, subCluster_comm, final_comm);
	return MPI_SUCCESS;
}

/*Function Prototype for Preproccessing*/
int getNodeID(char *name); // Takes csewsX as input and returns X
int getSubClusterID(int nodeID); // Takes X as input and uses Cluster Info Matrix to get the clusterID

/*Function Prototype for Actual Simulation*/
void simulate_Bcast(int argc, char* argv[], int num_of_doubles);
void simulate_Reduce(int argc, char* argv[],int num_of_doubles);
void simulate_Gather(int argc, char* argv[],int num_of_doubles);

int main(int argc, char* argv[])
{
	int D, len, num_of_doubles;
	D = atoi(argv[1]); /*gets the  value of D, the data size from Command Line*/
	int choice = atoi(argv[2]); /*gets the choice value which decides which Collective Call this Program will Simulate*/
	num_of_doubles = (D*1024)/8;

	if(choice==0)
		simulate_Bcast(argc, argv, num_of_doubles);
	else if(choice==1)
		simulate_Reduce(argc, argv, num_of_doubles);
	else if(choice==2)
		simulate_Gather(argc, argv, num_of_doubles);

	return 0;
}

void simulate_Bcast(int argc, char* argv[],int num_of_doubles){
	int len;
	char name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);

	/*Collecting rank, size, Processor Name, NodeID and subClusterID of the current Process*/
	int global_rank, global_size, subclusterID, nodeID;
	MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &global_size);
	MPI_Get_processor_name(name, &len);
	nodeID = getNodeID(name);
	subclusterID = getSubClusterID(nodeID);
	
	/*
	Creating a Sub Communicator based on nodeID. So, that only Ranks on Same Host/Node can communicate
	*/
	MPI_Comm sameNodeComm;
	MPI_Comm_split(MPI_COMM_WORLD, nodeID , global_rank, &sameNodeComm);

	int sameHost_rank;
	MPI_Comm_rank(sameNodeComm, &sameHost_rank);
	/*
	Creating a Sub Communicator based on ClusterID. So, that only nodes in same sub-cluster can communicate
	*/
	MPI_Comm sameSubClusterComm;
	MPI_Comm_split(MPI_COMM_WORLD, subclusterID  , global_rank, &sameSubClusterComm);
	/*
	Creating a Sub Communicator based on ClusterID. So, that only rank 0 processes of each Host present in same sub-cluster can communicate. This is done by further splitting sameSubClusterComm.
	*/


	MPI_Comm sameSubClusterComm_onlyFirstRanks;
	MPI_Comm_split(sameSubClusterComm, (sameHost_rank==0) ? 0: MPI_UNDEFINED, global_rank, &sameSubClusterComm_onlyFirstRanks);

	/*
	Creating a Sub Communicator so that only rank 0 processes of each sub-cluster can communicate. This is done by further splitting sameSubClusterComm_onlyFirstRanks.
	*/


	MPI_Comm final_comm = MPI_COMM_NULL;
	int samesubCluster_rank = -1;

	if(sameSubClusterComm_onlyFirstRanks!=MPI_COMM_NULL){
		MPI_Comm_rank(sameSubClusterComm_onlyFirstRanks, &samesubCluster_rank); 
	}
	MPI_Comm_split(MPI_COMM_WORLD , (samesubCluster_rank==0) ? 0: MPI_UNDEFINED, global_rank, &final_comm);

	/*Creating space for two 1D arrays*/
	double *array,*recv_array;
	array = (double*)malloc(num_of_doubles*sizeof(double));
	recv_array = (double*)malloc(num_of_doubles*sizeof(double));

	initialize_randomly(num_of_doubles, array, global_rank);

	double start_time, end_time, total_time, max_time_normal, max_time_optim;

	/***********************************
	The MPI_Bcast and MPI_Bcast_optimized Comparison
	************************************/	
	int i = 0;
	for (i = 0; i < 5 ; ++i)
	{
		initialize_randomly(num_of_doubles, array, global_rank);
		/*Simulation For MPI_Bcast()*/
		MPI_Barrier(MPI_COMM_WORLD);
		
		start_time = MPI_Wtime();
		MPI_Bcast(array, num_of_doubles, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		end_time = MPI_Wtime();
		total_time  = (end_time - start_time);
		MPI_Reduce (&total_time, &max_time_normal, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		max_time_normal = max_time_normal/5;
		/*Simulation For MPI_Bcast_optimized()*/

		//Inserting an MPI_Barrier() to ensure unbiased timing records.
		MPI_Barrier(MPI_COMM_WORLD);
		
		start_time = MPI_Wtime();
		MPI_Bcast_optimized(array, num_of_doubles, MPI_DOUBLE, 0 , MPI_COMM_WORLD, sameNodeComm, sameSubClusterComm_onlyFirstRanks, final_comm);
		end_time = MPI_Wtime();
		total_time  =  (end_time - start_time);
		MPI_Reduce (&total_time, &max_time_optim, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		max_time_optim = max_time_optim/5;
	}

	if (global_rank == 0)
	{
		printf("%.6lf\n", max_time_normal);// This is average value as I have divided by 5 earlier
		printf("%.6lf\n", max_time_optim);// This is average value as I have divided by 5 earlier
	}	

	/*Freeing memory*/
	free(array);
	free(recv_array);

	MPI_Finalize();

}
void simulate_Reduce(int argc, char* argv[],int num_of_doubles){
	int len;
	char name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);

	/*Collecting rank, size, Processor Name, NodeID and subClusterID of the current Process*/
	int global_rank, global_size, subclusterID, nodeID;
	MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &global_size);
	MPI_Get_processor_name(name, &len);
	nodeID = getNodeID(name);
	subclusterID = getSubClusterID(nodeID);
	
	/*
	Creating a Sub Communicator based on nodeID. So, that only Ranks on Same Host/Node can communicate
	*/
	MPI_Comm sameNodeComm;
	MPI_Comm_split(MPI_COMM_WORLD, nodeID , global_rank, &sameNodeComm);

	int sameHost_rank;
	MPI_Comm_rank(sameNodeComm, &sameHost_rank);
	/*
	Creating a Sub Communicator based on ClusterID. So, that only nodes in same sub-cluster can communicate
	*/
	MPI_Comm sameSubClusterComm;
	MPI_Comm_split(MPI_COMM_WORLD, subclusterID  , global_rank, &sameSubClusterComm);
	/*
	Creating a Sub Communicator based on ClusterID. So, that only rank 0 processes of each Host present in same sub-cluster can communicate. This is done by further splitting sameSubClusterComm.
	*/


	MPI_Comm sameSubClusterComm_onlyFirstRanks;
	MPI_Comm_split(sameSubClusterComm, (sameHost_rank==0) ? 0: MPI_UNDEFINED, global_rank, &sameSubClusterComm_onlyFirstRanks);

	/*
	Creating a Sub Communicator so that only rank 0 processes of each sub-cluster can communicate. This is done by further splitting sameSubClusterComm_onlyFirstRanks.
	*/


	MPI_Comm final_comm = MPI_COMM_NULL;
	int samesubCluster_rank = -1;

	if(sameSubClusterComm_onlyFirstRanks!=MPI_COMM_NULL){
		MPI_Comm_rank(sameSubClusterComm_onlyFirstRanks, &samesubCluster_rank); 
	}
	MPI_Comm_split(MPI_COMM_WORLD , (samesubCluster_rank==0) ? 0: MPI_UNDEFINED, global_rank, &final_comm);

	/*Creating space for two 1D arrays*/
	double *array,*recv_array;
	array = (double*)malloc(num_of_doubles*sizeof(double));
	recv_array = (double*)malloc(num_of_doubles*sizeof(double));

	initialize_randomly(num_of_doubles, array, global_rank);

	double start_time, end_time, total_time, max_time_normal, max_time_optim;

	/***********************************
	The MPI_Reduce and MPI_Reduce_optimized Comparison
	************************************/	
	int i = 0;
	for (i = 0; i < 5 ; ++i)
	{
		initialize_randomly(num_of_doubles, array, global_rank);
		/*Simulation For MPI_Reduce()*/
		MPI_Barrier(MPI_COMM_WORLD);
		start_time = MPI_Wtime();

		MPI_Reduce(array, recv_array, num_of_doubles, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

		end_time = MPI_Wtime();
		total_time  = (end_time - start_time);
		MPI_Reduce (&total_time, &max_time_normal, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		max_time_normal = max_time_normal/5;
		/*Simulation For MPI_Reduce_optimized()*/

		//Inserting an MPI_Barrier() to ensure unbiased timing records.
		MPI_Barrier(MPI_COMM_WORLD);
		
		start_time = MPI_Wtime();
		MPI_Reduce_optimized(array, recv_array, num_of_doubles,  MPI_DOUBLE, MPI_MIN, 0 , MPI_COMM_WORLD, sameNodeComm, sameSubClusterComm_onlyFirstRanks, final_comm);
		end_time = MPI_Wtime();
		total_time  =  (end_time - start_time);
		MPI_Reduce (&total_time, &max_time_optim, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		max_time_optim = max_time_optim/5;
	}

	if (global_rank == 0)
	{
		printf("%.6lf\n", max_time_normal); // These are average times as I have divided them by 5
		printf("%.6lf\n", max_time_optim);
	}	

	/*Freeing memory*/
	free(array);
	free(recv_array);

	MPI_Finalize();

}
void simulate_Gather(int argc, char* argv[],int num_of_doubles){
	int len;
	char name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);

	/*Collecting rank, size, Processor Name, NodeID and subClusterID of the current Process*/
	int global_rank, global_size, subclusterID, nodeID;
	MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &global_size);
	MPI_Get_processor_name(name, &len);
	nodeID = getNodeID(name);
	subclusterID = getSubClusterID(nodeID);
	
	/*
	Creating a Sub Communicator based on nodeID. So, that only Ranks on Same Host/Node can communicate
	*/
	MPI_Comm sameNodeComm;
	MPI_Comm_split(MPI_COMM_WORLD, nodeID , global_rank, &sameNodeComm);

	int sameHost_rank;
	MPI_Comm_rank(sameNodeComm, &sameHost_rank);
	/*
	Creating a Sub Communicator based on ClusterID. So, that only nodes in same sub-cluster can communicate
	*/
	MPI_Comm sameSubClusterComm;
	MPI_Comm_split(MPI_COMM_WORLD, subclusterID  , global_rank, &sameSubClusterComm);
	/*
	Creating a Sub Communicator based on ClusterID. So, that only rank 0 processes of each Host present in same sub-cluster can communicate. This is done by further splitting sameSubClusterComm.
	*/


	MPI_Comm sameSubClusterComm_onlyFirstRanks;
	MPI_Comm_split(sameSubClusterComm, (sameHost_rank==0) ? 0: MPI_UNDEFINED, global_rank, &sameSubClusterComm_onlyFirstRanks);

	/*
	Creating a Sub Communicator so that only rank 0 processes of each sub-cluster can communicate. This is done by further splitting sameSubClusterComm_onlyFirstRanks.
	*/


	MPI_Comm final_comm = MPI_COMM_NULL;
	int samesubCluster_rank = -1;

	if(sameSubClusterComm_onlyFirstRanks!=MPI_COMM_NULL){
		MPI_Comm_rank(sameSubClusterComm_onlyFirstRanks, &samesubCluster_rank); 
	}
	MPI_Comm_split(MPI_COMM_WORLD , (samesubCluster_rank==0) ? 0: MPI_UNDEFINED, global_rank, &final_comm);

	/*Creating space for two 1D arrays*/
	double *array,*recv_array;
	array = (double*)malloc(num_of_doubles*sizeof(double));
	recv_array = (double*)malloc(num_of_doubles*sizeof(double));

	initialize_randomly(num_of_doubles, array, global_rank);

	double start_time, end_time, total_time, max_time_normal, max_time_optim;
	double *big_receive_buffer;
	big_receive_buffer = (double*)malloc(num_of_doubles * global_size * 3 * sizeof(double));
	

	

	/***********************************
	The MPI_Gather and MPI_Gather_optimized Comparison
	************************************/	
	int i = 0;
	for (i = 0; i < 5 ; ++i)
	{
		initialize_randomly(num_of_doubles, array, global_rank);
		/*Simulation For MPI_Gather()*/
		MPI_Barrier(MPI_COMM_WORLD);
		
		start_time = MPI_Wtime();
		MPI_Gather(array, num_of_doubles, MPI_DOUBLE, big_receive_buffer, num_of_doubles, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		end_time = MPI_Wtime();
		
		total_time  = (end_time - start_time);
		MPI_Reduce (&total_time, &max_time_normal, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		max_time_normal = max_time_normal/5;
		/*Simulation For MPI_Gather_optimized()*/

		//Inserting an MPI_Barrier() to ensure unbiased timing records.
		MPI_Barrier(MPI_COMM_WORLD);
		
		start_time = MPI_Wtime();
		MPI_Gather_optimized( array, num_of_doubles, MPI_DOUBLE, big_receive_buffer, num_of_doubles, MPI_DOUBLE, 0, num_of_doubles*global_size ,MPI_COMM_WORLD, sameNodeComm, sameSubClusterComm_onlyFirstRanks, final_comm);

		end_time = MPI_Wtime();
		total_time  =  (end_time - start_time);
		MPI_Reduce (&total_time, &max_time_optim, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		max_time_optim = max_time_optim/5;
	}

	if (global_rank == 0)
	{
		printf("%.6lf\n", max_time_normal);// These are average times
		printf("%.6lf\n", max_time_optim);
	}	

	/*Freeing memory*/
	free(array);
	free(recv_array);
	free(big_receive_buffer);
	MPI_Finalize();

}

/*Auxilliary Functions*/
void initialize_randomly(int N, double *matrix, int rank){
	 srand(time(0)*(rank+1));
	 int i = 0;
	 for(i = 0; i< N; i++)
	 		matrix[i] = (rank+1)*((double)rand()/ RAND_MAX);
}
void display_array(int N, double *arr){
	printf("\n");
	int i;
	for (i = 0; i < N; ++i)
		printf("%.0lf ", arr[i]);
	printf("\n");
}
int getNodeID(char* name)
{
	char *end_ptr;
	return (int)strtol(name+5,&end_ptr,10);
}
int getSubClusterID(int nodeID)
{
	int i = 0;
	int j = 0;
	for (i = 0; i < 6; ++i){
		for (j = 0; j < 17; ++j){
			if(clusterInfo[i][j]==nodeID){
				return i;
			}
		}
	}
}
/*Optimized Collective Calls*/
int MPI_Bcast_optimized(void *buffer,int count,MPI_Datatype datatype,int root, MPI_Comm global_communicator,MPI_Comm sameHost_comm, MPI_Comm subCluster_comm, MPI_Comm final_comm){
	if(root!=0){
		/*Incase the 0-ranked process is not broadcasting, this makes the broadcasting rank to send the data to the 0 rank which then in turn broadcasts the data*/
		int global_rank;
		MPI_Comm_rank(global_communicator, &global_rank);
		if(global_rank==root){
			MPI_Send(buffer, count, datatype, 0, 0,global_communicator);
		}
		if(global_rank==0){
			MPI_Recv(buffer, count, datatype, root,0,global_communicator, MPI_STATUS_IGNORE);
		}
	}
	if(final_comm != MPI_COMM_NULL){
		MPI_Bcast(buffer, count, datatype, 0, final_comm);
	}

	if(subCluster_comm != MPI_COMM_NULL){
		MPI_Bcast(buffer, count, datatype, 0, subCluster_comm);
	}
	MPI_Bcast(buffer, count, datatype, 0, sameHost_comm);
	return MPI_SUCCESS;
}
int MPI_Reduce_optimized(void* send_buffer, void* receive_buffer, int count, MPI_Datatype datatype, MPI_Op operation,int root, MPI_Comm global_communicator,MPI_Comm sameHost_comm, MPI_Comm subCluster_comm, MPI_Comm final_comm){


	MPI_Reduce(send_buffer, receive_buffer, count, datatype, operation, 0, sameHost_comm);

	if(subCluster_comm != MPI_COMM_NULL){
		MPI_Reduce(receive_buffer, send_buffer, count, datatype, operation, 0, subCluster_comm);
	}

	if(final_comm != MPI_COMM_NULL){
		MPI_Reduce(send_buffer, receive_buffer, count, datatype, operation, 0, final_comm);
		
	}
	if(root!=0){
		/*Incase the 0-ranked process is not where the reduction needs to go, this code block makes the 0-rank to send the data to the destination */
		int global_rank;
		MPI_Comm_rank(global_communicator, &global_rank);
		if(global_rank==0){
			MPI_Send(receive_buffer, count, datatype, root, 0,global_communicator);
		}
		if(global_rank==root){
			MPI_Recv(receive_buffer, count, datatype, 0,0,global_communicator, MPI_STATUS_IGNORE);
		}
	}	
	return MPI_SUCCESS;
}
int MPI_Gather_optimized(double *sendbuf, int sendcount, MPI_Datatype sendtype, double *recvbuf, int recvcount,MPI_Datatype recvtype, int root, int param, MPI_Comm global_communicator,MPI_Comm sameHost_comm, MPI_Comm subCluster_comm, MPI_Comm final_comm){

	MPI_Gather(sendbuf, sendcount, sendtype, recvbuf+2*param, recvcount, recvtype, 0, sameHost_comm);

	int rank,size1, size2;
	//MPI_Barrier(MPI_COMM_WORLD);
	if(subCluster_comm != MPI_COMM_NULL){
		MPI_Comm_rank(subCluster_comm, &rank);
		MPI_Comm_size(sameHost_comm, &size1);
		if(rank!=0){
			MPI_Gather(recvbuf+2*param, size1*sendcount, sendtype, recvbuf+param, 2*recvcount, recvtype, 0, subCluster_comm);
		}else
		{
			MPI_Gather(recvbuf+2*param, size1*sendcount, sendtype, recvbuf+param, size1*recvcount, recvtype, 0, subCluster_comm);
		}
	}
	//MPI_Barrier(MPI_COMM_WORLD);

	if(final_comm != MPI_COMM_NULL){
		MPI_Comm_rank(final_comm, &rank);
		MPI_Comm_size(subCluster_comm, &size2);
		
		if(rank!=0){
			MPI_Gather(recvbuf+param, size1*size2*sendcount, sendtype, recvbuf, 4*recvcount, recvtype, 0, final_comm);
		}else
		{
			MPI_Gather(recvbuf+param, size1*size2*sendcount, sendtype, recvbuf, size1*size2*recvcount, recvtype, 0, final_comm);
		}

	}
	if(root!=0){
		/*Incase the 0-ranked process is not where data has to gather, this code block makes the 0-rank to send the data to the gathering destination */
		int global_rank;
		MPI_Comm_rank(global_communicator, &global_rank);
		if(global_rank==0){
			MPI_Send(recvbuf, param, recvtype, root, 0,global_communicator);
		}
		if(global_rank==root){
			MPI_Recv(recvbuf, param, recvtype, 0,0,global_communicator, MPI_STATUS_IGNORE);
		}
	}	
	return MPI_SUCCESS;

}
