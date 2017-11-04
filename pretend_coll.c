#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>

#define stripe_size 1048576
#define procs_per_node 24

int main(int argc, char **argv)
{
	int rank,nproc;
	int offset, len, end_offset, count, total_size;
	int *offset_array, *endoffset_array;
	double start, end, phase1, phase2, phase3, phase4, phase5, totaltime;
	int color;
	int i,j, nodes, send_agg_id, agg_id=-1;
	int sum, sumcount, agg_sum;
	char *buffer, *bigbuffer;
	char *filename;
	MPI_File fh;
	int *recv_size;
    MPI_Request *requests;
    MPI_Status *statuses;
    MPI_Comm agg_comm;

	//init
    MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    nodes = nproc/procs_per_node;

	//rank 0 will check and obtain parameters, block size will be bcasted to all processes
	if(rank == 0)
	{
		if(argc < 3)
		{
			fprintf(stderr, "You should provide me a file name and a block size\n");
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		len = strlen(argv[1]);
        //send the size to everyone so they can allocate space for the filename
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
		filename = (char *) malloc(len+1);
		strcpy(filename, argv[1]);
        //send the filename to everyone
        MPI_Bcast(filename, len+1, MPI_CHAR, 0, MPI_COMM_WORLD);
		len = atoi(argv[2]);
        //check the block size.
        //Also for simplicity, I've just implemented the first iteration, where each aggregator will write up to stripe size to the file system.
        total_size = len*nproc;
        if(total_size > (nodes*stripe_size))
        {
            len = (nodes*stripe_size)/nproc;
            total_size = len*nproc;
            fprintf(stderr, "Warning: the provided length was too big, it has been adjusted to %d.\n", len);
        }
        //send the information to everyone
		MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
		printf("Starting execution with %d processes, each of them will write %d bytes to the shared file %s\n", nproc, len, filename);
	}
	else
    {
	    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        filename = malloc(len+1);
        MPI_Bcast(filename, len+1, MPI_CHAR, 0, MPI_COMM_WORLD);
	    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
	    total_size = len*nproc; //we calculate it here because rank 0 already calculated it before the bcast.
    }

    //everyone will check if they are aggregators or not
    if((rank % procs_per_node) == 0) // I'm assuming ranks 0 to 23 are in node 0, 24 to 47 in rank 1, ...
    {
        //my rank could be an aggregator, but it depends on the size of the writes
        agg_id = rank/procs_per_node; //my id among the aggregators
        if(total_size < (agg_id*stripe_size))
            agg_id = -1; //I'm not an aggregator after all
//        else
  //          printf("I'm rank %d and I'm an aggregator\n", rank);
    }

	//fill test information and prepare write buffer
	offset = rank * len;
	end_offset = offset + len;
	count = 1;
	buffer = malloc(len);
	if(!buffer)
	{
		printf("PANIC! Cannot allocate memory\n");
		MPI_Abort(MPI_COMM_WORLD,1);
	}
	srand(rank);
	for(i=0; i< len; i++)
		buffer[i] = (char) (rand() % 40);
//	printf("Process %d - offset from %d to %d, len %d\n", rank, offset, end_offset, len);
	//allocate memory for the test
	offset_array = malloc(sizeof(int)*nproc);
	endoffset_array = malloc(sizeof(int)*nproc);
    requests = malloc(sizeof(MPI_Request)*(nproc*2 + 2));
    statuses = malloc(sizeof(MPI_Status)*(nproc*2 + 2));
	if((!offset_array) || (!endoffset_array) || (!requests) || (!statuses))
	{
		printf("PANIC! Cannot allocate memory\n");
		MPI_Abort(MPI_COMM_WORLD,1);
	}
	for(i = 0; i< nproc; i++)
	{
		offset_array[i] = 0;
		endoffset_array[i] = 0;
	}

	//1. two mpi_allgather calls so everybody knows starting and ending offsets
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
	MPI_Allgather(&offset, 1, MPI_INT, offset_array, 1, MPI_INT, MPI_COMM_WORLD);
	MPI_Allgather(&end_offset, 1, MPI_INT, endoffset_array, 1, MPI_INT, MPI_COMM_WORLD);
	end = MPI_Wtime();
	phase1 = end - start;
	MPI_Reduce(&phase1, &totaltime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(rank == 0)
		printf("PHASE 1 - %e (for me it was %e)\n", totaltime, phase1);
	//verify operations worked correctly
	if(rank == 0)
	{
		for(i=0; i< nproc; i++)
		{
			if((endoffset_array[i] - offset_array[i]) != len)
			{
				printf("Warning! Something is wrong, allgather did not result in what we were expecting... [%d]: %d %d\n", i, offset_array[i], endoffset_array[i]);
			}
		}
	}

    //aggregators will see which processes will send data to them
    if(agg_id >= 0)
    {
        recv_size = malloc(sizeof(int)*nproc);
        if(!recv_size)
        {
            fprintf(stderr, "PANIC! Could not allocate memory!\n");
            MPI_Abort(MPI_COMM_WORLD,1);
        }
        j=0;
        agg_sum=0;
        for(i=0; i<nproc; i++)
        {
            if(((offset_array[i] / stripe_size) % nodes) == agg_id) //actually the operation will not be perfect when we have multiple aggregators, as with 240 processes it is impossible to write 10MB (if all processes write the same amount of data). Because the stripe size is not divisible by 24, we'll have rank 24 sending to aggregator 0 and so on. The first aggregator will receive more data than stripe_size. To fix it, we would have to implement each process sending data to more than one aggregator, but i dont think it will change things enough to become important in our analysis 
            {
                recv_size[i] = len;
                agg_sum+=len;
                j+=1;
               // printf("I'm rank %d, aggregator %d and I'll receive from %d.\n", rank, agg_id, i);
            }
            else
                recv_size[i] = 0;
        }
//	printf("rank %d, aggregator %d, will receive from %d processes, a total of %d bytes\n", rank, agg_id, j, agg_sum);
        //allocate write buffer
        bigbuffer = malloc(j*len);
        if(!bigbuffer)
        {
            fprintf(stderr, "Could not allocate memory!\n");
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }
    //everyone will see to whom they must send their data
    send_agg_id = (offset / stripe_size) % nodes;
   // printf("I'm rank %d and I'll send to aggregator %d, rank %d.\n", rank, send_agg_id, send_agg_id*procs_per_node);
    send_agg_id = send_agg_id *procs_per_node; //we need the rank, not the id among the aggregators


	//2. two mpi_allreduce calls so everybody knows number of accesses and size of the whole thing
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
	MPI_Allreduce(&count, &sumcount, 1,  MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&len, &sum, 1,  MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	end = MPI_Wtime();
	phase2 = end - start;
	MPI_Reduce(&phase2, &totaltime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(rank == 0)
		printf("PHASE 2 - %e (for me it was %e)\n", totaltime, phase2);
	//verify operations worked
	if(rank == 0)
	{
		if((sumcount != nproc) || (sum != nproc*len))
		{
			printf("Warning! Something is wrong, allreduce did not worked as expected... %d %d\n", sumcount, sum);
		}
	}

	//3. send + recv between aggregator and process to communicate offset and len of the access
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
    j = 0;
	if(agg_id >= 0) //recv
	{
		for(i=0; i< nproc; i++)
		{
	            if(recv_size[i] > 0) 
        	    {
    			MPI_Irecv(&(offset_array[i]), 1, MPI_INT, i, i*2, MPI_COMM_WORLD, requests + j);
	    		MPI_Irecv(&(endoffset_array[i]), 1, MPI_INT, i, i*2 + 1, MPI_COMM_WORLD, requests + j + 1);
                	j+= 2;
		    }
	    	}
    	}
	//send
	MPI_Isend(&offset, 1, MPI_INT, send_agg_id, rank*2, MPI_COMM_WORLD, requests + j);
	MPI_Isend(&end_offset, 1, MPI_INT, send_agg_id, rank*2 + 1, MPI_COMM_WORLD, requests + j + 1);
	j += 2;
    	//wait
    	MPI_Waitall(j, requests, statuses);
	end = MPI_Wtime();
	phase3 = end - start;
	MPI_Reduce(&phase3, &totaltime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(rank == 0)
		printf("PHASE 3 - %e (for me it was %e)\n", totaltime, phase3);
	//verify operations worked correctly
	if(agg_id >= 0)
	{
		for(i=0; i< nproc; i++)
		{
			if((endoffset_array[i] - offset_array[i]) != len)
			{
				printf("Warning! Something is wrong, send+recv did not result in what we were expecting... [%d]: %d %d\n", i, offset_array[i], endoffset_array[i]);
			}
		}
    }
    for(i=0; i< j; i++)
    {
        if(statuses[i].MPI_ERROR != MPI_SUCCESS)
            printf("rank %d, found error in request %d", rank, i);
    }


	//4. send + recv between aggregator and process to exchange data
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
    j=0;
    //recv
	if(agg_id >= 0)
	{
		for(i = 0; i < nproc; i++)
		{
            if(recv_size[i] > 0)
            {
			    MPI_Irecv(bigbuffer + j*len, len, MPI_BYTE, i, i+(nproc*2)+1, MPI_COMM_WORLD, &(requests[j]));
                j++;
		    }
	    }
    }
    //send
	MPI_Isend(buffer, len, MPI_BYTE, send_agg_id, rank+(nproc*2)+1, MPI_COMM_WORLD, &(requests[j]));
    j++;
    //wait
    MPI_Waitall(j, requests, statuses);
	end = MPI_Wtime();
	phase4 = end - start;
	MPI_Reduce(&phase4, &totaltime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(rank == 0)
		printf("PHASE 4 - %e (for me it was %e)\n", totaltime, phase4);
    for(i=0; i< j; i++)
    {
        if(statuses[i].MPI_ERROR != MPI_SUCCESS)
            printf("rank %d, found error in request %d", rank, i);
    }

    //we'll need an communicator just for us aggregators
    color =  (agg_id >= 0) ? 0 : 1;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &agg_comm);
	//5. Aggregators will do I/O. We will not consider open and close times
	if(agg_id >= 0)
	{
		MPI_File_open(agg_comm, filename, MPI_MODE_CREATE|MPI_MODE_DELETE_ON_CLOSE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
//       		printf("Rank %d, aggregator %d will write %d bytes to file starting at offset %d\n", rank, agg_id, (agg_sum <= stripe_size) ? agg_sum : stripe_size, agg_id*stripe_size);
		start = MPI_Wtime();
		MPI_File_write_at(fh, agg_id*stripe_size, bigbuffer, (agg_sum <= stripe_size) ? agg_sum : stripe_size, MPI_BYTE, MPI_STATUS_IGNORE);
		end = MPI_Wtime();
		phase1 = end - start;
		start = MPI_Wtime();
		MPI_File_sync(fh);
		end = MPI_Wtime();
		phase2 = end - start;
		MPI_Reduce(&phase1, &phase5, 1, MPI_DOUBLE, MPI_MAX, 0, agg_comm);
	        MPI_Reduce(&phase2, &totaltime, 1, MPI_DOUBLE, MPI_MAX, 0, agg_comm);
		if(rank == 0)
			printf("PHASE 5 - %e + %e for sync (total %e)\n", phase5, totaltime, phase5 + totaltime);
		MPI_File_close(&fh);
	}


	//free memory
	if(offset_array)
		free(offset_array);
	if(endoffset_array)
		free(endoffset_array);
	if(buffer)
		free(buffer);
    if(requests)
        free(requests);
    if(statuses)
        free(statuses);
    if(filename)
        free(filename);
	if(agg_id > 0)
	{
		if(bigbuffer)
			free(bigbuffer);
		if(recv_size)
			free(recv_size);
	}
    MPI_Comm_free(&agg_comm);

	MPI_Finalize();
	return 0;
}
