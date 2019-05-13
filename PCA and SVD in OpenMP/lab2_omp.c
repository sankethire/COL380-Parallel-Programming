#include <malloc.h>
#include <math.h>
#include <omp.h>

// static float D_HAT_GLB[100000000];
static float U_calc[100000000];
static float VT_calc[100000000];

#define Abs(a)				  ( (a)>0.0?  (a) : (-(a)) ) 
#define SIGN(a,b)			   ( (b)>=0.0 ? Abs(a) : -Abs(a) )

struct Compare {int value; int pos;};
#pragma omp declare reduction(maximum : struct Compare : omp_out = omp_in.value > omp_out.value ? omp_in : omp_out)

static double normcal(double a, double b)
{
	double a_abs = fabs(a), b_abs = fabs(b), c_abs, r;

	if (a_abs > b_abs)	   
	{ 
		c_abs = b_abs / a_abs; 
		r = a_abs * sqrt(1.0 + c_abs * c_abs); 
	}
	else if (b_abs > 0.0) 
	{ 
		c_abs = a_abs / b_abs;
		r = b_abs * sqrt(1.0 + c_abs * c_abs); 
	}
	else r = 0.0;
	return r;
}

static double max(double a, double b)
{
	if (a > b) return a;
	else return b;
}


// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void SVD_calc(int M, int N, float* D, float** U, float** SIGMA, float** V_T)
{  
    // printf("this is SVD\n");
    float D_mat[M][N];
    // printf("\nD_mat0 = \n");
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {   
            D_mat[i][j] = D[i*N + j];
            // printf("%f ", D_mat[i][j]);
        }   
        // printf("\n");
    }

		float D_mat1[M][N];
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            D_mat1[i][j] = D_mat[i][j];
        }
    }


    float D_transpose[N][M];
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < M; j++)
        {
            D_transpose[i][j] = D_mat[j][i];
        }
    }

    // printf("\nD_transpose = \n");
    // for(int i = 0; i < N; i++)
    // {
    //     for(int j = 0; j < M; j++)
    //     {
    //         printf("%f ",D_transpose[i][j]);
    //     }
    //     printf("\n");
    // }



	

    int pp = 0;

	int button, i, ts, j, jj, k, ql, num;
	double qc, qf, qh, qs, qx, qy, qz;
	double normofD_mat = 0.0, qg = 0.0, factor = 0.0;
    double *qrv;
    float v_svd[N][N];
    float wtrack[N][2];
		// int wis[N];
    float eigenvals[N];
  
	
	qrv = (double*)malloc((unsigned int) N*sizeof(double));

	for (i = 0; i < N; i++) {
		ql = i + 1;
		qrv[i] = factor * qg;
		qg = qs = factor = 0.0;
		if (i < M) {
			for (k = i; k < M; k++) 
				factor += fabs((double)D_mat[k][i]);
			if (factor) {
				for (k = i; k < M; k++) {
					D_mat[k][i] = (float)((double)D_mat[k][i]/factor);
					qs += ((double)D_mat[k][i] * (double)D_mat[k][i]);
				}
				qf = (double)D_mat[i][i];
				qg = -SIGN(sqrt(qs), qf);
				qh = qf * qg - qs;
				D_mat[i][i] = (float)(qf - qg);
				if (i != N - 1) {
					for (j = ql; j < N; j++) {
						for (qs = 0.0, k = i; k < M; k++) 
							qs += ((double)D_mat[k][i] * (double)D_mat[k][j]);
						qf = qs / qh;
						for (k = i; k < M; k++) 
							D_mat[k][j] += (float)(qf * (double)D_mat[k][i]);
					}
				}
				for (k = i; k < M; k++) 
					D_mat[k][i] = (float)((double)D_mat[k][i]*factor);
			}
		}
		eigenvals[i] = (float)(factor * qg);
	
		qg = qs = factor = 0.0;
		if (i < M && i != N - 1) {
			for (k = ql; k < N; k++) 
				factor += fabs((double)D_mat[i][k]);
			if (factor) {
				for (k = ql; k < N; k++) {
					D_mat[i][k] = (float)((double)D_mat[i][k]/factor);
					qs += ((double)D_mat[i][k] * (double)D_mat[i][k]);
				}
				qf = (double)D_mat[i][ql];
				qg = -SIGN(sqrt(qs), qf);
				qh = qf * qg - qs;
				D_mat[i][ql] = (float)(qf - qg);
				for (k = ql; k < N; k++) 
					qrv[k] = (double)D_mat[i][k] / qh;
				if (i != M - 1) {
					for (j = ql; j < M; j++) {
						for (qs = 0.0, k = ql; k < N; k++) 
							qs += ((double)D_mat[j][k] * (double)D_mat[i][k]);
						for (k = ql; k < N; k++) 
							D_mat[j][k] += (float)(qs * qrv[k]);
					}
				}
				for (k = ql; k < N; k++) 
					D_mat[i][k] = (float)((double)D_mat[i][k]*factor);
			}
		}
		normofD_mat = max(normofD_mat, (fabs((double)eigenvals[i]) + fabs(qrv[i])));
	}
  
	for (i = N - 1; i >= 0; i--) {
		if (i < N - 1) {
			if (qg) {
				for (j = ql; j < N; j++)
					v_svd[j][i] = (float)(((double)D_mat[i][j] / (double)D_mat[i][ql]) / qg);
				for (j = ql; j < N; j++) 
				{
					for (qs = 0.0, k = ql; k < N; k++) 
						qs += ((double)D_mat[i][k] * (double)v_svd[k][j]);
					for (k = ql; k < N; k++) 
						v_svd[k][j] += (float)(qs * (double)v_svd[k][i]);
				}
			}
			for (j = ql; j < N; j++) 
				v_svd[i][j] = v_svd[j][i] = 0.0;
		}
		v_svd[i][i] = 1.0;
		qg = qrv[i];
		ql = i;
	}
  
	for (i = N - 1; i >= 0; i--) {
		ql = i + 1;
		qg = (double)eigenvals[i];
		if (i < N - 1) 
			for (j = ql; j < N; j++) 
				D_mat[i][j] = 0.0;
		if (qg) {
			qg = 1.0 / qg;
			if (i != N - 1) {
				for (j = ql; j < N; j++) {
					for (qs = 0.0, k = ql; k < M; k++) 
						qs += ((double)D_mat[k][i] * (double)D_mat[k][j]);
					qf = (qs / (double)D_mat[i][i]) * qg;
					for (k = i; k < M; k++) 
						D_mat[k][j] += (float)(qf * (double)D_mat[k][i]);
				}
			}
			for (j = i; j < M; j++) 
				D_mat[j][i] = (float)((double)D_mat[j][i]*qg);
		} else {
			for (j = i; j < M; j++) 
				D_mat[j][i] = 0.0;
		}
		++D_mat[i][i];
	}

	for (k = N - 1; k >= 0; k--) { 
		for (ts = 0; ts < 30; ts++) {
			button = 1;
			for (ql = k; ql >= 0; ql--) {
				num = ql - 1;
				if (fabs(qrv[ql]) + normofD_mat == normofD_mat) {
					button = 0;
					break;
				}
				if (fabs((double)eigenvals[num]) + normofD_mat == normofD_mat) 
					break;
			}
			if (button) {
				qc = 0.0;
				qs = 1.0;
				for (i = ql; i <= k; i++) {
					qf = qs * qrv[i];
					if (fabs(qf) + normofD_mat != normofD_mat) {
						qg = (double)eigenvals[i];
						qh = normcal(qf, qg);
						eigenvals[i] = (float)qh; 
						qh = 1.0 / qh;
						qc = qg * qh;
						qs = (- qf * qh);
						for (j = 0; j < M; j++) {
							qy = (double)D_mat[j][num];
							qz = (double)D_mat[j][i];
							D_mat[j][num] = (float)(qy * qc + qz * qs);
							D_mat[j][i] = (float)(qz * qc - qy * qs);
						}
					}
				}
			}
			// printf("\neigenvalue is %f\n",(eigenvals[k] * eigenvals[k]) );

			wtrack[k][0] = eigenvals[k];
			wtrack[k][1] = pp;
			pp++;
			qz = (double)eigenvals[k];
			if (ql == k) { 
				if (qz < 0.0) {
					eigenvals[k] = (float)(-qz);
					for (j = 0; j < N; j++) 
						v_svd[j][k] = (-v_svd[j][k]);
				}
				break;
			}
			if (ts >= 30) {
				free((void*) qrv);
			}
	
			qx = (double)eigenvals[ql];
			num = k - 1;
			qy = (double)eigenvals[num];
			qg = qrv[num];
			qh = qrv[k];
			qf = ((qy - qz) * (qy + qz) + (qg - qh) * (qg + qh)) / (2.0 * qh * qy);
			qg = normcal(qf, 1.0);
			qf = ((qx - qz) * (qx + qz) + qh * ((qy / (qf + SIGN(qg, qf))) - qh)) / qx;
		  
			qc = qs = 1.0;
			for (j = ql; j <= num; j++) {
				i = j + 1;
				qg = qrv[i];
				qy = (double)eigenvals[i];
				qh = qs * qg;
				qg = qc * qg;
				qz = normcal(qf, qh);
				qrv[j] = qz;
				qc = qf / qz;
				qs = qh / qz;
				qf = qx * qc + qg * qs;
				qg = qg * qc - qx * qs;
				qh = qy * qs;
				qy = qy * qc;
				for (jj = 0; jj < N; jj++) {
					qx = (double)v_svd[jj][j];
					qz = (double)v_svd[jj][i];
					v_svd[jj][j] = (float)(qx * qc + qz * qs);
					v_svd[jj][i] = (float)(qz * qc - qx * qs);
					// printf("\nvsvd = jj j = %f,	 jj i = %f\n",v_svd[jj][j],v_svd[jj][i]);
				}
				qz = normcal(qf, qh);
				eigenvals[j] = (float)qz;
				if (qz) {
					qz = 1.0 / qz;
					qc = qf * qz;
					qs = qh * qz;
				}
				qf = (qc * qg) + (qs * qy);
				qx = (qc * qy) - (qs * qg);
				for (jj = 0; jj < M; jj++) {
					qy = (double)D_mat[jj][j];
					qz = (double)D_mat[jj][i];
					D_mat[jj][j] = (float)(qy * qc + qz * qs);
					D_mat[jj][i] = (float)(qz * qc - qy * qs);
				}
			}
			// printf("l --> qrv[%d] = %lf\n",l,qrv[l] );
			qrv[ql] = 0.0;
			qrv[k] = qf;
			// printf("k --> qrv[%d] = %lf\n",k,f );
			eigenvals[k] = (float)qx;
		}
	}

   

	// printf(">>>>>>\n");
	// // for(int i = 0; i < M-1; ++i){
	// 	for (int j = 0; j < N; ++j)
	// 	{
	// 		printf("%f ", qrv[j]);
	// 	}
	// 	printf("\n");
	// printf(">>>>>>\n");
		
		
	free((void*) qrv);

    float ww[M];
    //sort eigens
	float tmp;

	 #pragma omp parallel 
	 {
	 for (int i = 0; i < N; ++i) 
        {
						struct Compare maxi;
						maxi.value = eigenvals[i];
						maxi.pos = i;

						#pragma omp parallel for reduction(maximum:maxi)
            for (int j = i + 1; j < N; ++j)
            {
 
                // if (eigenvals[i] < eigenvals[j])
								if (eigenvals[j] > maxi.value) 
                {
									 maxi.value = eigenvals[j];
									 maxi.pos = j; 
                    // tmp =  eigenvals[i];
                    // eigenvals[i] = eigenvals[j];
                    // eigenvals[j] = tmp;
 
                }
 
            }
						tmp =  eigenvals[i];
            eigenvals[i] = eigenvals[maxi.pos];
            eigenvals[maxi.pos] = tmp;
 					
 
        }
	 }		

	// for (int i = 0; i < N; ++i)
	// {
	// 	printf("%f\n", (eigenvals[i]*eigenvals[i]));
	// }

	int h = 0;
	
	// printf("\n");

		for (int i = N-1; i >= 0; i--)
		{	
			// printf("wtrack = %f  and i = %d, j = %d\n", wtrack[i][0],i,0);
			ww[h++] = wtrack[i][0];
			// wis[i] = i;
		}
		
		
	// printf("\n");

	// for (int i = 0; i < N; ++i)
	// {
	// 		printf("%f ",ww[i] );
	// }

	// 	printf("\n");

	// 	printf("\n");

	// for (int i = 0; i < N; ++i)
	// {
	// 		printf("wis = %d", wis[i] );
	// }

	// 	printf("\n");

int worder[N];
h = 0;
	for (int i = 0; i < N; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				/* code */
				if(eigenvals[i] == ww[j]){
					worder[h++] = j;;
				}
			}
		}	


// for (int i = 0; i < N; ++i)
// 	{
// 		// for (int j = 0; j < m; ++j)
// 		// {
// 			printf(" worder = %d ",worder[i] );
// 		// }
// 	}
// 		printf("\n");

	int ord = N-1;
	int worder1[N];
	for(int i = 0; i < N; i++)
	{
		worder1[ord] = worder[i];
		ord--;
	}
	
	
	// for (int i = 0; i < N; ++i)
	// {
	// 	// for (int j = 0; j < m; ++j)
	// 	// {
	// 		printf(" worder1 = %d ",worder1[i] );
	// 	// }
	// }
	

	// 	printf("\n\n");

  //  printf("-----------------------------------------------------\n\n");		

// printf("D = \n");
// 	for (int i = 0; i < M; ++i)
// 	{
// 		for (int j = 0; j < N; ++j)
// 		{
// 			printf("%f ", D_mat1[i][j]);
// 		}
// 		printf("\n");
// 	}		
// 	printf("\n");


	// D_mat1[]


float vorder[N][N];
float vorderT[N][N];
for (int i = 0; i < N; ++i)
{
	for (int j = 0; j < N; ++j)
	{
		/* code */
		// vorder[i][j] = v_svd[worder[i]][j];
		vorder[i][j] = v_svd[worder1[i]][j];
	}
}

for(int i = 0; i < N; i++)
{
	vorder[i][0] = -vorder[i][0];
}


// printf("V = \n");
// for (int i = 0; i < N; ++i)
// 	{
// 		for (int j = 0; j < N; ++j)
// 		{
// 			printf("%f ",vorder[i][j] );
// 		}
// 		printf("\n");
// 	}

// 		printf("\n");
	

for (int i = 0; i < N; ++i)
{
	for (int j = 0; j < N; ++j)
	{
		vorderT[i][j] = vorder[j][i];
	}
}

printf("V_T = \n");
for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			printf("%f ",vorderT[i][j] );
		}
		printf("\n");
	}

		printf("\n");
	
float sigma1[M][N];
float sigmaI[N][M];

for (int i = 0; i < N; ++i)
{
	for (int j = 0; j < M; ++j)
	{
		if(i == j){
			// sigma1[i][j] = (eigenvals[i]);
			sigmaI[i][j] = 1/((eigenvals[i]));
		}
		else{
			// sigma1[i][j] = 0;
			sigmaI[i][j] = 0;
		}
	}
}

for (int i = 0; i < M; ++i)
{
	for (int j = 0; j < N; ++j)
	{
		if(i == j){
			sigma1[i][j] = (eigenvals[i]);
			// sigmaI[i][j] = 1/((eigenvals[i]));
		}
		else{
			sigma1[i][j] = 0;
			// sigmaI[i][j] = 0;
		}
	}
}
// printf("sigma = \n");
// for (int i = 0; i < M; ++i)
// 	{
// 		for (int j = 0; j < N; ++j)
// 		{
// 			printf("%f ",sigma1[i][j] );
// 		}
// 		printf("\n");
// 	}

// 		printf("\n");

// printf("sigma_I = \n");
// for (int i = 0; i < N; ++i)
// 	{
// 		for (int j = 0; j < M; ++j)
// 		{
// 			printf("%f ",sigmaI[i][j] );
// 		}
// 		printf("\n");
// 	}

// 		printf("\n");

	// multiply(int m1, int m2, int mat1[][m2], 
 //            int n1, int n2, int mat2[][n2],int res[m1][m2]) 	

	 float dv_mat[M][N];
	 // multiply(m,n,amn,n,n,v,dv_mat);
	 float dvsi[M][M];
	 // multiply(m,n,dv_mat,n,n,sigmaI,dvsi);

		float dtd[N][N];
		float sumu = 0;

		#pragma omp parallel private(i,j,k) shared(sumu)
		{
			#pragma omp for schedule(guided)
			for(int i = 0; i < N; i++)
			{
				for(int j = 0; j < N; j++)
				{
					for(int k = 0; k < M; k++)
					{
						sumu += D_transpose[i][k] * D_mat1[k][j];
					}
					dtd[i][j] = sumu;
					sumu = 0;
				}
				
			}
		}
// printf("\ndtd = \n");
// for(int i = 0; i < N; i++)
// 		{
// 			for(int j = 0; j < N; j++)
// 			{
// 				printf("%f ",dtd[i][j]);
// 			}
// 			printf("\n");
// 		}
// 			printf("\n");
		

		


	float sum = 0;
#pragma omp parallel private(i,j,k) shared(sum)
{
	 #pragma omp for schedule(guided)
	 for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
          sum = sum + D_mat1[i][k]*vorder[k][j];
        }
 
        dv_mat[i][j] = sum;
        sum = 0;
      }
    }
}	
//   printf("dv_mat = \n");
// for (int i = 0; i < M; ++i)
// 	{
// 		for (int j = 0; j < N; ++j)
// 		{
// 			printf("%f ",dv_mat[i][j] );
// 		}
// 		printf("\n");
// 	}

// 		printf("\n");  

	sum = 0;
	
	 for (int i = 0; i < M; i++) {
      for (int j = 0; j < M; j++) {
        for (int k = 0; k < N; k++) {
          sum = sum + dv_mat[i][k]*sigmaI[k][j];
        }
 
        dvsi[i][j] = sum;
        sum = 0;
      }
    }
//      printf("U / V' = \n");
	 
// for (int i = 0; i < M; ++i)
// 	{
// 		for (int j = 0; j < M; ++j)
// 		{
// 			printf("%f ",dvsi[i][j] );
// 		}
// 		printf("\n");
// 	}

// 		printf("\n"); 

	float usigma[M][N];

	sum = 0;
	 for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        for (int k = 0; k < M; k++) {
          sum = sum + dvsi[i][k]*sigma1[k][j];
        }
 
        usigma[i][j] = sum;
        sum = 0;
      }
    }

//     printf("usvt1 = \n");
// for (int i = 0; i < M; ++i)
// 	{
// 		for (int j = 0; j < N; ++j)
// 		{
// 			printf("%f ",usigma[i][j] );
// 		}
// 		printf("\n");
// 	}

// 		printf("\n"); 

    float usvt[M][N];
    sum = 0;
		
	 for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
          sum = sum + usigma[i][k]*vorderT[k][j];
        }
 
        usvt[i][j] = sum;
        sum = 0;
      }
    }

//     printf("usvt2 = \n");
// for (int i = 0; i < M; ++i)
// 	{
// 		for (int j = 0; j < N; ++j)
// 		{
// 			printf("%f ",usvt[i][j] );
// 		}
// 		printf("\n");
// 	}

// 		printf("\n"); 
 
	// printf("\ncheck wala V_T = \n");	
	#pragma omp parallel
	{
		#pragma omp parallel for
		for(int i = 0; i < N; i++)
		{
			for(int j = 0; j < N; j++)
			{
				*(*(V_T) + i*N + j) = vorderT[i][j];
				// printf("%f ",*(*(V_T) + i*N + j));
			}
			// printf("\n");
		}
	}
			// printf("\n");

//  *(*(sigma) + i *n + j)
		#pragma omp parallel
		{
				#pragma omp parallel for
				for(int i = 0; i < M; i++)
			{
				for(int j = 0; j < M; j++)
				{
					*(*(U) + i*N + j) = dvsi[i][j];
				}
			}
		}
    
// 			for(int i = 0; i < M; i++)
// 		{
// 			for(int j = 0; j < N; j++)
// 			{
// 				printf("%f ",*(*(U)+i*N+j));
// 			}
// 			printf("\n");
// 		}

		// printf("\nsigma** = \n");
		

		for(int i = 0; i < N; i++)
		{
				*(*(SIGMA) + i) = sigma1[i][i];
				// printf("%f ",*(*(SIGMA) + i));
				
		}

// 		printf("\n\n");

	
		

		
}

void transpose(float* a, float* b, int m, int n) {
    #pragma omp parallel 
		{
			#pragma omp parallel for
			for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            *(a+i*n+j) = *(b+j*m+i);
        }
    	}
		}
}

void SVD(int M, int N, float* D, float** U, float** SIGMA, float** V_T)
{
		float* U_new = U_calc;
    float* VT_new = VT_calc;
    SVD_calc(M, N, D, &U_new, SIGMA, &VT_new);
		transpose(*U, VT_new, N, N);
    transpose(*V_T, U_new, M, M);
		
}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void PCA(int retention, int M, int N, float* D, float* U, float* SIGMA, float** D_HAT, int *K)
{	
		// printf("\nThis is PCA\n");
		float sum = 0;
		float sig_mat[N];
    // for(int i = 0; i < N; i++)
		// {
		// 	for(int j = 0; j < N; j++)
		// 	{
		// 		sig_mat[i][j] = SIGMA[i*N+j] ;
		// 		sum += sig_mat[i][j];
		// 		printf("%f ",sig_mat[i][j]);
		// 	}
		// 	printf("\n");
		// }
		
		float ks[N];
		for(int i = 0; i < N; i++)
		{
			ks[i] = (*(SIGMA + i)) * (*(SIGMA + i));
			// printf("%f ",*(SIGMA + i));
			sum += ks[i];
		}

		float divks[N];
		float ss = 0;
		int cntk;

		// printf("\ndivks = \n");
		for(int i = 0; i < N; i++)
		{
			ss += ks[i]/sum;
			// printf("%f ", ks[i]/sum);
			// printf("sum[%d] = %f ,",i,ss*100);
			if(ss*100>=retention){
				cntk = i+1;
				break;
			}
		}
		if(retention == 100){
			cntk = N;
		}
		

	// 	float sumk = 0;
	// // float retention_val;
	// 	for(int i = 0; i < N; i++)
	// 	{
	// 		sumk += ks[i];
	// 		cntk = i;
	// 		// printf("\n(sumk*100)/sum = %f\n", (sumk*100)/sum);
	// 		if((sumk*100)/sum >= retention){
	// 			// retention_val = sumk;
	// 			break;
	// 		}
	// 	}

		// cntk = cntk+1;
		// printf("\nsum = %f\n",sum);
		// printf("\nsum/k = %f\n",sum/cntk);
		*(K) = cntk;
		printf("\nk = %d\n", *(K));
		


		// for(int i = 0; i < N; i++)
		// {
		// 	ks[i] = SIGMA[i];
		// 	sum += ks[i];
		// 	printf("%f ",ks[i]);
		// }
		// printf("\n");

	// printf("\nsum = %f\n",sum);

	// float D_mat[M][N];
	// for(int i = 0; i < M; i++)
	// {
	// 	for(int j = 0; j < N; j++)
	// 	{
	// 		D_mat[i][j] = D[i*N+j];
	// 	}
		
	// }
	
	

	// printf("\n\n");
	
	//////////////////////////////////////////////////////////////////////
	
	

	*D_HAT=(float*)malloc(sizeof(float) *M*cntk);

	printf("\nU in PCA = \n");
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			printf("%f ",U[i*N+j]);
		}
		printf("\n");
	}

	printf("\n");

	float wihati[N][N];
	// printf("\nWIHATI = \n");
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			wihati[i][j] = U[i*N+j];
			// printf("wihati[%d][%d] = %f ",i,j,wihati[i][j]);

		}
		// printf("\n");
	}
	

	float wmat[N][cntk];
	// printf("\nW = \n");
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < cntk; j++)
		{
			wmat[i][j] = wihati[i][j];
			// printf("%f ",wmat[i][j]);
		}
		// printf("\n");
	}
	
	float dhat[M][cntk];
	float dsum = 0;
	for(int i = 0; i < M; i++)
	{
		for(int j = 0; j < cntk; j++)
		{
			for(int k = 0; k < N; k++)
			{
				dsum += D[i*N+k]*wmat[k][j];
			}
			dhat[i][j] = dsum;
			dsum = 0;
		}
		
	}
	// printf("\ndhat = \n");
	// for(int i = 0; i < M; i++)
	// {
	// 	for(int j = 0; j < cntk; j++)
	// 	{
	// 		printf("%f ",dhat[i][j]);
	// 	}
	// 	printf("\n");
	// }
	
	//////////////////////////////////////////////////////////////////////

	// float d_dihati[M][N];
  //   sum = 0;
	//  for (int i = 0; i < M; i++) {
  //     for (int j = 0; j < N; j++) {
  //       for (int k = 0; k < N; k++) {
  //         sum = sum + usigma[i][k]*vorderT[k][j];
  //       }
 
  //       usvt[i][j] = sum;
  //       sum = 0;
  //     }
  //   }

// printf("\nU in PCA is = \n");
// for(int i = 0; i < M; i++)
// {
// 	for(int j = 0; j < N; j++)
// 	{
// 		printf("%f ",U[i*N+j]);
// 	}
// 	printf("\n");
// }

// float* temp_DHAT = D_HAT_GLB;

#pragma omp parallel
{
	#pragma omp parallel for
	for(int i = 0; i < M; i++)
	{
		for(int j = 0; j < cntk; j++)
		{
			*(*(D_HAT) + i*cntk+j) = dhat[i][j];
		}
		
	}
}

// *D_HAT = temp_DHAT;

// printf("\nD_HAT = \n");
// for(int i = 0; i < M; i++)
// {
// 	for(int j = 0; j < cntk; j++)
// 	{
// 		printf("%f ", *(*(D_HAT)+i*cntk+j));
// 	}
// 	printf("\n");
// }




// *D_HAT=(float *)malloc(sizeof(float) * (M*cntk) ); // M rows
// for(int i=0;i<M;i++)
// {
//     *(D_HAT+i)=(float*)malloc(sizeof(float)*cntk);
// }


// 	for(int i = 0; i < M; i++)
// 	{
// 		for(int j = 0; j < cntk; j++)
// 		{


// 			*(*D_HAT + i*cntk + j) = dihati[i][j];


// 		}
// 	}

// 	// printf("hjh\n");

// 	for(int i = 0; i < M; i++)
// 	{
// 		for(int j = 0; j < cntk; j++)
// 		{
// 			printf("%f ",*(*(D_HAT) + i*cntk + j) );
// 		}
// 		printf("\n");
// 	}

	

}
