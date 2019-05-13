#include "lab3_cuda.h"
#include <bits/stdc++.h>
using namespace std;

#define TOLERANCE 0.001
#define JACOBI_UPDATE_TOLERANCE 0.001

__global__ void Matrix_Multiplication_Cuda(double* A, int Am, int An, double* B, int Bm, int Bn, double* C, int Cm, int Cn) {

    
    __shared__ double A_shared[16][16];
    __shared__ double B_shared[16][16];
    
    int m = blockIdx.y*16 + threadIdx.y;
    int n = blockIdx.x*16 + threadIdx.x;
    double mul_c = 0;
      for (int i = 0; i < (16 + An - 1)/16; i++) {
  
        if (i*16 + threadIdx.x < An && m < Am) {
            A_shared[threadIdx.y][threadIdx.x] = A[m*An + i*16 + threadIdx.x];
        }
        else {
            A_shared[threadIdx.y][threadIdx.x] = 0.0;
        }
  
        if (i*16 + threadIdx.y < Bm && n < Bn)  {
            B_shared[threadIdx.y][threadIdx.x] = B[(i*16 + threadIdx.y)*Bn + n];
        }
        else {
            B_shared[threadIdx.y][threadIdx.x] = 0.0;
        }
  
        __syncthreads();
  
        for (int n = 0; n < 16; ++n) {
            mul_c += A_shared[threadIdx.y][n] * B_shared[n][threadIdx.x];
        }
  
        __syncthreads();
  
    }
  
    if (m < Cm && n < Cn) {
        C[ ((blockIdx.y * blockDim.y + threadIdx.y)*Cn) + (blockIdx.x*blockDim.x) + threadIdx.x ] = mul_c;
    }
  
  }

double* mat_transpose(double* A, int Am, int An) {
    double* A_T;
    A_T = (double*)malloc(__SIZEOF_POINTER__*An*Am);
    for(int i = 0; i < An; i++){
        for(int j = 0 ; j < Am; j++){
            A_T[i*Am+j] = A[j*An+i];
        }
    } 
    return A_T;
}

void print_matrix(double* A, int Am, int An) {
   for(int i = 0; i < Am; i++){
       for(int j = 0 ; j < An; j++){
           printf("%f ", A[i*An+j]);
       }
       printf("\n");
   }
}

void print_vector(double* A, int An) {
    printf("[");
    for(int i=0; i<An-1; i++)
        printf("%f,",A[i]);
    printf("%f]\n",A[An-1]);
}

void print_mat(double** A, int Am, int An) {
    printf("[");
    for (int i=0; i<Am; i++){
        if (i>0)
            printf(" ");
        printf("[");
        for (int j=0; j<An-1; j++){
            printf("%f, ",A[i][j]);
        }
        if (i < Am-1)
            printf("%f]\n",A[i][An-1]);
    }
    printf("%f]]\n",A[Am-1][An-1]);
}


double* mat_mul(double* A, int Am, int An, 
                 double* B, int Bm, int Bn){
    double *C;
    C = (double*)malloc(__SIZEOF_POINTER__*Am*Bn);

    for (int i=0; i<Am; i++){
        for (int j=0; j<Bn; j++){
            C[i*Bn+j] = 0;
            for (int k=0; k<An; k++){
                C[i*Bn+j] += A[i*An + k] * B[k*Bn + j];
            }
        }
    }

    return C;
}

double** mult(double** A, int Am, int An, 
                 double** B, int Bm, int Bn){
    double **C;
    C = (double**)malloc(__SIZEOF_POINTER__*Am);
    for (int i=0; i<Am; i++)
        C[i] = (double*)malloc(__SIZEOF_DOUBLE__*Bn);

    for (int i=0; i<Am; i++){
        for (int j=0; j<Bn; j++){
            C[i][j] = 0;
            for (int k=0; k<An; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

double *S; //Symmetric matrix(input)
int N_jacobi; 
double  *e; //eigenvalues
double **E; //eigenvectors
int  *ind;
bool *changed;
int  state;

int maxind(int k) {
    int m = k+1;
    for (int i = k+2; i < N_jacobi; i++){
        if (fabs(S[k*N_jacobi + i]) > fabs(S[k*N_jacobi + m])){
            m = i;
        }
    }
    return m;
}

void update(int k, double t) {
    double ek_prev = e[k];
    e[k] = ek_prev + t;

    if (e[k] < 0) e[k] = 0;

    if (changed[k] && fabs(ek_prev - e[k]) < JACOBI_UPDATE_TOLERANCE) {
        changed[k] = false;
        state = state - 1;
    }
    else if ((! changed[k]) && fabs(ek_prev - e[k]) > JACOBI_UPDATE_TOLERANCE) {
        changed[k] = true;
        state = state + 1;
    }
}

void rotate(int k, int l, int i, int j, double c, double s,
            bool eigenvectors){
    double** mat1;
    double** mat2;
    double** mat3;

    mat1 = (double**)malloc(__SIZEOF_POINTER__*2);
    mat1[0] = (double*)malloc(__SIZEOF_DOUBLE__*2);
    mat1[1] = (double*)malloc(__SIZEOF_DOUBLE__*2);
    mat1[0][0] = c; mat1[0][1] = -s;
    mat1[1][0] = s; mat1[1][1] = c;

    mat2 = (double**)malloc(__SIZEOF_POINTER__*2);
    mat2[0] = (double*)malloc(__SIZEOF_DOUBLE__*1);
    mat2[1] = (double*)malloc(__SIZEOF_DOUBLE__*1);
    if (eigenvectors){
        mat2[0][0] = E[i][k];
        mat2[1][0] = E[i][l];
    }
    else {
        mat2[0][0] = S[k*N_jacobi + l];
        mat2[1][0] = S[i*N_jacobi + j];
    }

    mat3 = mult(mat1, 2, 2, mat2, 2, 1);

    if (eigenvectors){
        E[i][k] = mat3[0][0];
        E[i][l] = mat3[1][0];
    }
    else{
        S[k*N_jacobi + l] = mat3[0][0];
        S[i*N_jacobi + j] = mat3[1][0];
    }

    free(mat1[0]);
    free(mat1[1]);
    free(mat1);
    free(mat2[0]);
    free(mat2[1]);
    free(mat2);
    free(mat3[0]);
    free(mat3[1]);
    free(mat3);
}

void init_jacobi() {
    E = (double**)malloc(__SIZEOF_POINTER__*N_jacobi);
    for (int i=0; i<N_jacobi; i++){
        E[i] = (double*)malloc(__SIZEOF_DOUBLE__*N_jacobi);
        for (int j=0; j<N_jacobi; j++){
            E[i][j] = 0;
        }
        E[i][i] = 1;
    }
    state = N_jacobi;

    e = (double*)malloc(__SIZEOF_DOUBLE__*N_jacobi);
    ind = (int*)malloc(__SIZEOF_INT__*N_jacobi);
    changed = (bool*)malloc(sizeof(bool)*N_jacobi);

    for (int k=0; k<N_jacobi; k++){
        ind[k]     = maxind(k);
        e[k]       = S[k*N_jacobi + k];
        changed[k] = true;
    }
}

void Jacobi(double* input_matrix, int n, 
            double** eigenvalues, double*** eigenvectors) {
    N_jacobi = n;
    S = input_matrix;
    init_jacobi();

    while(state != 0){
        int m = 0;

        for (int k=1; k<N_jacobi-1; k++){
            if (fabs(S[k*N_jacobi + ind[k]]) > fabs(S[m*N_jacobi + ind[m]])){
                m = k;
            }
        }

        int k = m;
        int l = ind[m];
        double p = S[k*N_jacobi + l];
        double y = (e[l] - e[k]) / 2.0;
        double d = fabs(y) + sqrt(p*p + y*y);
        double r = sqrt(p*p + d*d);
        double c = d / r;
        double s = p / r;
        double t = (p*p) / d;

        if (y < 0.0) { s = -s; t = -t; }

        S[k*N_jacobi + l] = 0.0;
        update(k, -t);
        update(l, t);

        for (int i=0; i<k; i++)  { rotate(i, k, i, l, c, s, false); }
        for (int i=k+1; i<l; i++){ rotate(k, i, i, l, c, s, false); }
        for (int i=l+1; i<N_jacobi; i++)  { rotate(k, i, l, i, c, s, false); }

        for (int i=0; i<N_jacobi; i++){
            rotate(k, l, i, i, c, s, true);
        }

        ind[k] = maxind(k);
        ind[l] = maxind(l);
    }

    *eigenvalues = e;
    *eigenvectors = E;

}            


// /*
//  *****************************************************
//      TODO -- You must implement this function
//  *****************************************************
// */
void SVD_and_PCA (int M, 
        int N, 
        double* D, 
        double** U, 
        double** SIGMA, 
        double** V_T, 
        int* SIGMAm,
        int* SIGMAn, 
        double** D_HAT, 
        int *K,
        int retention) {
    // write your code here
    
    double *eigenvalues, **eigenvectors;
    printf("\nD = \n");
    print_matrix(D,M,N);
    double* D_T = mat_transpose(D,M,N);
    printf("\nD_T = \n");
    print_matrix(D_T,N,M);

    double* D_D;
    double* DT_D;
    double* DTD_D;
    
    double* prod = (double*)malloc(__SIZEOF_POINTER__*N*N);

    cudaMalloc((void **)&D_D, sizeof(double)*M*N);
    cudaMalloc((void **)&DT_D, sizeof(double)*N*M);
    cudaMalloc((void **)&DTD_D, sizeof(double)*N*N);
    
    cudaMemcpy(D_D, D, sizeof(double)*M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(DT_D, D_T, sizeof(double)*N*M, cudaMemcpyHostToDevice);
    
    dim3 dimGrid0((N + 16 - 1) / 16, (N + 16 - 1) / 16);
    dim3 dimBlock0(16, 16);

    Matrix_Multiplication_Cuda<<<dimGrid0, dimBlock0>>>(DT_D,N,M, D_D,M,N, DTD_D,N,N);
    cudaMemcpy(prod, DTD_D, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
    cudaFree(D_D);
    cudaFree(DTD_D);

    // double* prod = mat_mul(D_T,N,M,D,M,N);
    printf("\nDT_D = \n");
    print_matrix(prod,N,N);

    Jacobi(prod, N, &eigenvalues, &eigenvectors);

    printf("\neigenvalues:\n");
    print_vector(eigenvalues, N);

    printf("\neigenvectors:\n");
    print_mat(eigenvectors, N, N);

    vector<pair<double,int>> eigenpairs;
    for(int i = 0; i < N; i++) {
        eigenpairs.push_back(make_pair(eigenvalues[i],i));
    }
    sort(eigenpairs.begin(),eigenpairs.end());
    reverse(eigenpairs.begin(),eigenpairs.end());
    

    printf("\nsorted eigenvalues = \n");
    for(int i = 0; i < N;i++) {
        printf("%f ",eigenpairs[i].first);
    }
    
    printf("\n\nindices sorted according eigenvalues = \n");
    for(int i = 0; i < N;i++) {
        printf("%d ",eigenpairs[i].second);
    }

    printf("\n");
    
    // for(int i = 0; i < N; i++) {
    //     for(int j = 0 ; j < N ; j++) {
    //         printf("%f ",eigenvectors[i][j]);
    //     }
    //     printf("\n");
    // }

    double sorted_eigenvectors[N][N];
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            sorted_eigenvectors[i][j] = eigenvectors[i][eigenpairs[j].second];
        }
    }

    printf("\nsorted eigenvectors = \n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%f ",sorted_eigenvectors[i][j]);
        }
        printf("\n");
    }

   double t_sorted_eigenvectors[N][N];
   for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            t_sorted_eigenvectors[i][j] = sorted_eigenvectors[j][i];
        }
    }

    printf("\nt_sorted eigenvectors = \n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%f ",t_sorted_eigenvectors[i][j]);
        }
        printf("\n");
    }
    
    
    double* inv_sigma_mat = (double*)malloc(__SIZEOF_POINTER__*N*M);
    double* U_transpose = (double*)malloc(__SIZEOF_POINTER__*N*N);
    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            inv_sigma_mat[i*M+j] = 0;
        }
    }

    *SIGMA = (double*)malloc(__SIZEOF_POINTER__*N);
    *U = (double*)malloc(__SIZEOF_POINTER__*N*N);

    for(int k=0; k<N; k++){
        int i = eigenpairs[k].second;
        (*SIGMA)[i] = (double)sqrt(eigenpairs[i].first);    
        inv_sigma_mat[i*N+i] = 1/(double)sqrt(eigenpairs[i].first); 
        
        for(int j=0; j<N; j++){
            U_transpose[i*N+j] = eigenvectors[j][k];
            (*U)[j*N + i] = eigenvectors[j][k];
        }
    }

    printf("\nU = \n");
    print_matrix(*U,N,N);


    // for(int i = 0; i < N; i++) {
    //     int p = eigenpairs[i].second;
    //     inv_sigma_mat[i*N+i] = 1/(double)sqrt(eigenpairs[i].first); 
    // }
    
    // printf("clear");


    // for(int i = 0; i < N; i++)
	// 	{
    //            (*SIGMA)[i] = (double)sqrt(eigenpairs[i].first);			
    //     }
    
    // for(int i = 0; i < N; i++)
	// 	{
    //            printf("sigmas = %f\n",(*SIGMA)[i]);			
	// 	}    

    double* inv_sigma_mat_D;
    double* U_transpose_D;
    double* V_transpose_D;
    double* prod1 = (double*)malloc(__SIZEOF_POINTER__*M*N);
    double* prod1_D;

    cudaMalloc((void **)&inv_sigma_mat_D, sizeof(double)*N*M);
    cudaMalloc((void **)&U_transpose_D, sizeof(double)*N*N);
    cudaMalloc((void **)&prod1_D, sizeof(double)*M*N);

    cudaMemcpy(inv_sigma_mat_D, inv_sigma_mat, sizeof(double)*N*M, cudaMemcpyHostToDevice);
    cudaMemcpy(U_transpose_D, U_transpose, sizeof(double)*N*M, cudaMemcpyHostToDevice);
    
    dim3 dimGrid1((M + 16 - 1) / 16, (N + 16 - 1) / 16);
    dim3 dimBlock1(16, 16);

    Matrix_Multiplication_Cuda<<<dimGrid1, dimBlock1>>>(inv_sigma_mat_D,M,N, U_transpose_D,N,N, prod1_D,M,N);
    cudaFree(inv_sigma_mat_D);
    cudaFree(U_transpose_D);
    
    cudaMalloc((void**)&V_transpose_D, sizeof(double)*M*M);
    dim3 dimGrid2((M + 16 - 1) / 16, (M + 16 - 1) / 16);
    dim3 dimBlock2(16, 16);
    
    *V_T = (double*)malloc(__SIZEOF_POINTER__*M*M);
    Matrix_Multiplication_Cuda<<<dimGrid2, dimBlock2>>>(prod1_D,M,N, DT_D,N,M, V_transpose_D,M,M);
    cudaMemcpy(*V_T, V_transpose_D, sizeof(double)*M*M, cudaMemcpyDeviceToHost);
    cudaFree(inv_sigma_mat_D);
    cudaFree(U_transpose_D);
    cudaFree(prod1_D);
    cudaFree(V_transpose_D);
    
    free(prod);
    free(prod1);
    free(inv_sigma_mat);
    free(U_transpose);

    printf("\nSVD done!\n");

    double eigensum = 0;
    for(int i=0; i<N; i++) {
        eigensum += eigenpairs[i].first;
    }
    printf("\nvariance = \n");
    double variance[N];
    for(int i=0; i<N; i++) {
        variance[i] = (eigenpairs[i].first)/eigensum;
        printf("%f ",variance[i]);
    }

    printf("\n");

    double travelsum = 0;
    int ncols = 0;
    for(int i = 0; i<N; i++) {
        printf("\ntravelsum = %f\n",travelsum);
        if((travelsum*100) < (double)retention){
            travelsum += variance[i];
            ncols++;
        } 
        else {
            break;
        }
    }

    *K = ncols;
    printf("\nK = (%d,%d)\n", ncols, *K);

    double* U_current = (double*)malloc(__SIZEOF_POINTER__*N*(*K));
    for(int i=0; i<N; i++){
        for(int j=0; j<ncols; j++){
            U_current[i*(ncols) + j] = (*U)[i*N + j];
        }
    }

    printf("\nD = \n");
    print_matrix(D,M,N);
    printf("\nU_current = \n");
    print_matrix(U_current,N,ncols);
    printf("\n\n");

    *D_HAT = (double*)malloc(__SIZEOF_DOUBLE__*M*ncols);
    double* D_Dest; 
    double* D_HAT_Dest; 
    double* U_Dest;
    cudaMalloc((void **)&D_Dest, __SIZEOF_POINTER__*M*N);
    cudaMalloc((void **)&U_Dest,__SIZEOF_DOUBLE__*N*(ncols));
    cudaMalloc((void **)&D_HAT_Dest, __SIZEOF_POINTER__*M*(ncols));
    cudaMemcpy(D_Dest, D, sizeof(double)*M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(U_Dest, U_current, sizeof(double)*N*ncols, cudaMemcpyHostToDevice);


    double* dd = (double*)malloc(__SIZEOF_POINTER__*M*N);
    double* ud = (double*)malloc(__SIZEOF_POINTER__*N*ncols);
    cudaMemcpy(dd, D_Dest, sizeof(double)*M*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(ud, U_Dest, sizeof(double)*N*ncols, cudaMemcpyDeviceToHost);

    printf("\nDD = \n");
    print_matrix(dd,M,N);
    printf("\nU_currentD = \n");
    print_matrix(ud,N,ncols);
    printf("\n\n");


    dim3 dimGrid3((M + 16 - 1) / 16, (M + 16 - 1) / 16);
    dim3 dimBlock3(16, 16);

    Matrix_Multiplication_Cuda<<<dimGrid3, dimBlock3>>>(D_Dest,M,N, U_Dest,N,ncols, D_HAT_Dest,M,ncols);
    cudaMemcpy(*D_HAT, D_HAT_Dest, sizeof(double)*M*ncols, cudaMemcpyDeviceToHost);

    printf("\nD_HAT\n");
    print_matrix(*D_HAT,M,ncols);

    printf("\nV_T = \n");
    print_matrix(*V_T,M,M);


    /*****************************************************************************************************************************************************/
                                                                /* U calculation */
    /*****************************************************************************************************************************************************/
    
    // double dv_mat[M][N];
    // double sum = 0;
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //       for (int k = 0; k < N; k++) {
    //         sum = sum + D[i*N+k]*sorted_eigenvectors[k][j];
    //       }
   
    //       dv_mat[i][j] = sum;
    //       sum = 0;
    //     }
    // }

    // double dvsi[M][M];
    // double sum1 = 0;
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < M; j++) {
    //       for (int k = 0; k < N; k++) {
    //         sum1 = sum1 + dv_mat[i][k]*inv_sigma_mat[k][j];
    //         // printf("\n(%d,%d,%d)\n",i,j,k);
    //       }
    //       dvsi[i][j] = sum1;
    //       sum1 = 0;
    //     }
    // }

    // // printf("\n\n%f\n\n",dvsi[0][0]);
    // printf("\nU = \n");
    // for(int i = 0; i < M; i++) {
    //     for(int j = 0; j < M; j++) {
    //         printf("%f ",dvsi[i][j]);
    //     }
    //     printf("\n");
    // }

    // /*****************************************************************************************************************************************************/


    // /*****************************************************************************************************************************************************/
    //                                                             /* Correctness Check */
    // /*****************************************************************************************************************************************************/
    // // usvt
    // double usigma[M][N];
    // double sum2 = 0;
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //       for (int k = 0; k < M; k++) {
    //         sum2 = sum2 + dvsi[i][k]*sigma_mat[k][j];
    //       }
   
    //       usigma[i][j] = sum2;
    //       sum2 = 0;
    //     }
    // }

    // double usvt[M][N];
    // double sum3 = 0;
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //       for (int k = 0; k < N; k++) {
    //         sum3 = sum3 + usigma[i][k]*t_sorted_eigenvectors[k][j];
    //       }
   
    //       usvt[i][j] = sum3;
    //       sum3 = 0;
    //     }
    // }

    // printf("\ncheck_mat = \n");
    // for(int i = 0; i < M; i++) {
    //     for(int j = 0; j < N; j++) {
    //         printf("%f ",usvt[i][j]);
    //     }
    //     printf("\n");
    // }

    // /*****************************************************************************************************************************************************/

    // double eigensum = 0;
    // for(int i=0; i<N; i++) {
    //     eigensum += eigenpairs[i].first;
    // }
    // printf("\nvariance = \n");
    // double variance[N];
    // for(int i=0; i<N; i++) {
    //     variance[i] = (eigenpairs[i].first)/eigensum;
    //     printf("%f ",variance[i]);
    // }

    // printf("\n");

    // double travelsum = 0;
    // int ncols = 0;
    // for(int i = 0; i<N; i++) {
    //     printf("\ntravelsum = %f\n",travelsum);
    //     if((travelsum*100) < retention){
    //         travelsum += variance[i];
    //         ncols++;
    //     } 
    //     else {
    //         break;
    //     }
    // }

    // printf("\nK = %d\n", ncols);

    // double wmat[N][ncols];
    // for(int i=0; i<N; i++) {
    //     for(int j=0; j<ncols; j++) {
    //         wmat[i][j] = sorted_eigenvectors[i][j];
    //     }
    // }

    // printf("\nW = \n");
    // for(int i = 0; i < N; i++) {
    //     for(int j = 0; j < ncols; j++) {
    //         printf("%f ",wmat[i][j]);
    //     }
    //     printf("\n");
    // }

    // double dhat[M][ncols];
    // double sum4 = 0;
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < ncols; j++) {
    //       for (int k = 0; k < N; k++) {
    //         sum4 = sum4 + D[i*N+k]*wmat[k][j];
    //       }
   
    //       dhat[i][j] = sum4;
    //       sum4 = 0;
    //     }
    // }

    // printf("\nD_HAT = \n");
    // for(int i = 0; i < M; i++) {
    //     for(int j = 0; j < ncols; j++) {
    //         printf("%f ",dhat[i][j]);
    //     }
    //     printf("\n");
    // }

}

