#include <mex.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

void mexFunction(int nargout, mxArray *argout[], int nargin, const mxArray *argin[]) {
	int n1, n2, d1, d2, i, j, k, m, d;
	double *A1, *A2, *h1, *h2, *B;
	n1 = mxGetM(argin[0]);
	d1 = mxGetN(argin[0]); 
	A1 = mxGetPr(argin[0]);
	h1 = mxGetPr(argin[1]);
	n2 = mxGetM(argin[2]);
	d2 = mxGetN(argin[2]);
	A2 = mxGetPr(argin[2]);
	h2 = mxGetPr(argin[3]);
	m = (int)mxGetScalar(argin[4]);
	d = d1 + d2 - 1;
	argout[0] = mxCreateDoubleMatrix(m, d, mxREAL);
	B = mxGetPr(argout[0]);
	memset(B, 0, sizeof(double) * m * d);
	
	double f1[8], f2[8];
	if (d2 == 1) {
		memset(f2, 0, sizeof(f2));
		for (i = 0; i < n2; ++i) f2[(int)h2[i] - 1] += A2[i];
		for (k = 0; k < d1; ++k) {
			memset(f1, 0, sizeof(f1));
			for (i = 0; i < n1; ++i) f1[(int)h1[i] - 1] += A1[k * n1 + i];
			for (i = 0; i < m; ++i)
				for (j = 0; j < m; ++j)
				B[k * m + (i + j) % m] += f1[i] * f2[j];
		}
	}
	else {
		memset(f1, 0, sizeof(f1));
		for (i = 0; i < n1; ++i) f1[(int)h1[i] - 1] += A1[i];
		for (k = 0; k < d2; ++k) {
			memset(f2, 0, sizeof(f2));
			for (i = 0; i < n2; ++i) f2[(int)h2[i] - 1] += A2[k * n2 + i];
			for (i = 0; i < m; ++i)
				for (j = 0; j < m; ++j)
				B[k * m + (i + j) % m] += f1[i] * f2[j];
		}
	}
}
