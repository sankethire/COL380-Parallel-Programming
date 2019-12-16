#include "lab4_mpi.h"

#include <malloc.h>
#include <string.h>
#include "mpi.h"


void set_pattern_array(int* arr,int processes,int patterns_per_process, int patterns) 
{
	for(int i = 0; i < processes; i++)
	{
		arr[i] = patterns_per_process;
	}
	int remnant = patterns % processes;
	if(remnant != 0) { arr[processes-1] = arr[processes-1] + remnant; }
}						

int* get_witness_array(int length, int period,char* current_pattern)
{
	int* witness_array = (int*)malloc(__SIZEOF_POINTER__*period);
	witness_array[0] = 0;
	for (int i = 0; i < period; i++)
	{
		for (int j = 0; j < length; j++)
		{
			if(current_pattern[j] != current_pattern[i+j])
			{
				witness_array[i] = j;
				break;
			}
		}
	}
	return witness_array;
}

int duel_index(char* z, char* y, int* phi,int i, int j, int n)
{
	int k = phi[j-i];
	if( (j+k >= n) || (z[j+k] != y[k]) ) { return i; }
	else { return j; }	
}

int is_pattern_at_index(char* pattern, char* text, int length)
{
	for (int i = 0; i < length; i++)
	{
		if(pattern[i] != text[i]) { return 0; }
	}
	return 1;
}

int* non_periodic_pattern_matching(int n, char* text, int length, int* witness_array, char* current_pattern)
{
	int index;
	int* match_positions = (int*)malloc(__SIZEOF_POINTER__* ((n-1)/((length+1)/2) + 1) );
	for(int i = 0; i < ((n-1)/((length+1)/2) + 1); i++)
	{
		int sz = (length+1)/2;
		if( i == (n-1)/((length+1)/2)) { sz = n % ((length+1)/2); }
		index = 0;
		for (int j = 0; j < sz; j++)
		{	
			int ii = index+i*((length+1)/2);
			int jj = j+i*((length+1)/2);
			index = duel_index(text,current_pattern,witness_array,ii,jj,n) - (i * ((length+1)/2) );
		}
		index = index + (i * ((length+1)/2) );
		match_positions[index] = -1;

		if(is_pattern_at_index(current_pattern, text+index, length)) { match_positions[i] = index; }
	}
	return match_positions;
}

int is_x_in_arr(int* arr, int n, int x){
	for (int i = 0; i < n; i++)
	{
		if(arr[i] == x) { return 1; }
	}
	return 0;
}


int prefix_index;

void pattern_matching_algorithm(int n, char *text,
								int *m_set,	int *p_set,	char **pattern_set, 
								int *pma_match_counts, int *pma_matches, int pat, 
								int p_mpi_rank, int patterns_per_process)
{
	char* txt = text;
	prefix_index = 0;
	for (int i = 0; i < pat; i++)
	{
		int ptr = i + (p_mpi_rank*patterns_per_process);

		int length1 = m_set[ptr];
		int period1 = p_set[ptr];
		char* current_pattern1 = pattern_set[ptr];
		int* witness_array1 = get_witness_array(length1, period1, current_pattern1);

		int length2 = 2*period1 - 1;
		int period2 = p_set[ptr];
		char* current_pattern2 = (char*)malloc(sizeof(char)*length2);
		memcpy(current_pattern2,current_pattern1,length2);
		int* witness_array2 = get_witness_array(length1, period1, current_pattern2);

		int* match_positions = non_periodic_pattern_matching(n,text,length2,witness_array2,current_pattern2);

		//u,k and v
		char* u = (char*)malloc(sizeof(char)*period1);
		//u = current pattern1
		memcpy(u, current_pattern1,period1);
		//k
		int k = length1/period1;
		//v
		char* v = (char*)malloc(sizeof(char)* (length1-k*period1));
		//v = current pattern(kp : m - 1)
		memcpy(v, current_pattern1 + k*period1, (length1-k*period1));

		// usquarev u^2v
		char* usquarev = (char*)malloc(sizeof(char) * (period1+period1+length1-k*period1));
		memcpy(usquarev, u, period1);
		memcpy(usquarev+period1, u, period1);
		memcpy(usquarev+period1+period1, v,length1-k*period1);

		int M[n];
		int q;
		for(int i=0; i<n; i++){
			M[i] = 0;
			if( is_x_in_arr(match_positions,((n-1)/((length2+1)/2) + 1), i) &&  is_pattern_at_index(usquarev, txt+i, period1+period1+length1-k*period1)) { M[i] = 1; }
		}

		int* C[period1];
		for(int i=0; i<period1; i++){
			int cnt = 1;
			while( i + cnt*period1 < n) { cnt++; }
			int S[cnt];
			for(int j=0; j<cnt; j++){
				S[j] = M[i+j*period1];
			}
			C[i] = (int*) malloc(__SIZEOF_POINTER__* cnt );
			for(int j=0; j+(k-2) < cnt; j++){
				C[i][j] = 0;
				for(q=0;q<k-1;q++){
					if(S[j+q] != 1) { break; }
				}
				if(q==k-1) { C[i][j] = 1; }
			}
		}
		int MATCH [n-length1+1];
		for(int i=0; i <= n-length1; i++){
			for(int j=0; j<period1; j++){
				if(i%period1 == j){
					q = i/period1;
					MATCH[i] = C[j][q];
				}
			}
		}
		int ti=0;
		for(int j=0; j <= n-length1; j++){
			if(MATCH[j]==1){
				pma_matches[ti+prefix_index] = j;
				ti++;
			}
		}

		pma_match_counts[i] = ti;
		prefix_index += ti;		
	}
}



// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void periodic_pattern_matching (
		int n, 
		char *text, 
		int num_patterns, 
		int *m_set, 
		int *p_set, 
		char **pattern_set, 
		int **match_counts, 
		int **matches)
{
	//Initialize the MPI environment
	// MPI_Init(NULL, NULL);

	int p_mpi_processes;
	MPI_Comm_size(MPI_COMM_WORLD, &p_mpi_processes);

	int p_mpi_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &p_mpi_rank);

	if(p_mpi_rank == 0)
	{
		*match_counts = (int*)malloc(__SIZEOF_INT__*num_patterns);
		*matches = (int*)malloc(__SIZEOF_INT__*num_patterns*n);
	}

	int patterns_per_process = (num_patterns / p_mpi_processes) ;
	int* pattern_array = (int*)malloc(__SIZEOF_POINTER__*p_mpi_processes);
	set_pattern_array(pattern_array, p_mpi_processes, patterns_per_process, num_patterns);
	int pat = pattern_array[p_mpi_rank];

	int* pma_match_counts = (int*)malloc(__SIZEOF_POINTER__*pat);
	int* pma_matches = (int*)malloc(__SIZEOF_POINTER__*pat*n);

	pattern_matching_algorithm( n, text,
								m_set,	p_set,	pattern_set, 
								pma_match_counts, pma_matches, pat, 
								p_mpi_rank, patterns_per_process);

	int tpma_match_counts[pat];
	for (int ai = 0; ai < pat; ai++)
	{
		tpma_match_counts[ai] = pma_match_counts[ai];
	}

	int tpma_matches[pat*n];
	for (int ai = 0; ai < pat*n; ai++)
	{
		tpma_matches[ai] = pma_matches[ai];
	}	
	// printf("ok here\n");
	if (p_mpi_rank > 0) {
		MPI_Send(tpma_match_counts, pat, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(tpma_matches, prefix_index, MPI_INT, 0, 0, MPI_COMM_WORLD);
		
	} 
	else{
		memcpy( (*match_counts), tpma_match_counts, __SIZEOF_INT__*pat);
		memcpy( (*matches), tpma_matches, __SIZEOF_INT__*prefix_index);
		free(pma_match_counts);
		free(pma_matches);
		int final_patterns = pat;
		int final_prefix_index = prefix_index;
		for(int li=1; li<p_mpi_processes; li++){
			int new_patterns = pattern_array[li];
			int new_pma_match_counts[new_patterns];
			MPI_Recv(new_pma_match_counts, new_patterns, MPI_INT, li, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			int new_prefix_index = 0;
			for(int lj =0; lj<new_patterns; lj++)
				new_prefix_index += new_pma_match_counts[lj];
			
			int new_pma_matches[new_prefix_index];
			MPI_Recv(new_pma_matches, new_prefix_index, MPI_INT, li, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			memcpy( (*match_counts) + final_patterns, new_pma_match_counts, __SIZEOF_INT__*new_patterns);
			memcpy( (*matches) + final_prefix_index, new_pma_matches, __SIZEOF_INT__*new_prefix_index);
			final_patterns += new_patterns;
			final_prefix_index += new_prefix_index;
		}
	}
	free(pattern_array);

	//Finalize the MPI environment
	// MPI_Finalize();
}
