%module QuadLinkForwardModel

%{
	#define SWIG_FILE_WITH_INIT
	#include "QuadLinkForwardModel.h"
%}

%include "numpy.i"

%init %{
	import_array();
%}

%apply (double* IN_ARRAY2, int DIM1, int DIM2){(double* xValues, int numSamples, int dim_states)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2){(double* u_t, int num_actions, int dim_actions)}
%apply double {dt}
%apply (double* IN_ARRAY1, int DIM1){(double* m_t, int dim_m)}
%apply (double* IN_ARRAY1, int DIM1){(double* l_t, int dim_l)}
%apply (double* IN_ARRAY1, int DIM1){(double* I_t, int dim_I)}
%apply double {G}
%apply (double* IN_ARRAY1, int DIM1){(double* k_t, int dim_k)}
%apply double {dst}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2){(double* outArray, int num_out, int dim_out)}

%include "QuadLinkForwardModel.h"