#include <mex.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

void mexFunction(int nargout, mxArray *argout[], int nargin, const mxArray *argin[]) {
	int n, i;
	double *A, x, sum = 0;
	n = mxGetM(argin[0]); 
	A = mxGetPr(argin[0]);
	x = mxGetScalar(argin[1]);
	for (i = 0; i < n && sum < x; sum += A[i++]);
	argout[0] = mxCreateDoubleScalar((double)i);
}
