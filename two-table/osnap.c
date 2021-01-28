#include <mex.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

void mexFunction(int nargout, mxArray *argout[], int nargin, const mxArray *argin[]) {
	int m, n, d, s, i, j, k;
	double *S, *sign, *A, *B, v, *pr;
	mwIndex *jc, *ir;
	n = mxGetM(argin[0]);
	d = mxGetN(argin[0]);
	m = (int)mxGetScalar(argin[1]);
	s = (int)mxGetScalar(argin[2]);
	S = mxGetPr(argin[3]);
	sign = mxGetPr(argin[4]);
	argout[0] = mxCreateDoubleMatrix(m, d, mxREAL);
	B = mxGetPr(argout[0]);
	memset(B, 0, m * d * sizeof(double));
	if (!mxIsSparse(argin[0])) {
		/* Dense */
		A = mxGetPr(argin[0]);
		for (j = 0; j < d; ++j)
			for (i = 0; i < n; ++i) {
				v = A[j * n + i];
				for (k = 0; k < s; ++k) 
					B[j * m + (int)(S[k * n + i]) - 1] += sign[k * n + i] * v;
			}
	} else {
		/* Sparse */
		pr = mxGetPr(argin[0]);
		ir = mxGetIr(argin[0]);
		jc = mxGetJc(argin[0]);
		for (j = 0; j < d; ++j)
			for (i = jc[j]; i < jc[j+1]; ++i) {
				v = pr[i];
				for (k = 0; k < s; ++k) 
					B[j * m + (int)(S[k * n + ir[i]]) - 1] += sign[k * n + ir[i]] * v;
			}
	}
}
