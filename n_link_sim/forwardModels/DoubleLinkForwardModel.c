#include "math.h"

#define SQ(x) ((x)*(x))

/*
 * double pendulum forward model
 * 2 modes: 
 * mode 1: simulate dt/dst (init 50) iterations -> returns state (phi1, dphi1, phi2, dphi2, phi3, dphi3, phi4, dphi4)
 * mode 2: one iteration (dt==dst) -> returns accelerations (ddphi1, ddphi2, ddphi3, ddphi4)
 *
 **/

void simulate(double *xValues, int numSamples, int dim_states,
              double *u_t, int num_actions, int dim_actions,
              double dt,
              double *m_t, int dim_m,
              double *l_t, int dim_l,
              double *I_t, int dim_I,
              double g,
              double *k_t, int dim_k,
              double dst,
              int use_gains,
              double *PDSetPoints, int dim_sp,
              double *PDGains, int dim_gains,
              double *outArray, int num_out, int dim_out)
{
    /*matlab calls: xdot = doublePendulum_C_ForwardModel(x, u, dt, m, l, I, g, k, dst);*/
    /*x,u,m,l, I are vectors. all other variables are scalars. */
    /* k denotes the friction coefficient and dst the simulation time step */
    
/*    double *xValues, *u_t, *l_t, *m_t, *I_t, *g_t, *k_t, *dt_t, *dst_t, *PDSetPoints, *PDGains;
    xValues = mxGetPr(prhs[0]);
    u_t = mxGetPr(prhs[1]);
    dt_t = mxGetPr(prhs[2]);
    m_t = mxGetPr(prhs[3]);
    l_t = mxGetPr(prhs[4]);
    I_t = mxGetPr(prhs[5]);
    g_t = mxGetPr(prhs[6]);
    k_t = mxGetPr(prhs[7]);
    dst_t = mxGetPr(prhs[8]);
    PDSetPoints = mxGetPr(prhs[9]);
    PDGains = mxGetPr(prhs[10]);
  */

  //  double dt = dt_t[0];
    double m1 = m_t[0];
    double m2 = m_t[1];
    double l1 = l_t[0];
    double l2 = l_t[1];
    double I1 = I_t[0];
	double I2 = I_t[1];
    
   // double g = g_t[0];
    double VISCOUS_FRICTION1 = k_t[0];
    double VISCOUS_FRICTION2 = k_t[1];
  //  double dst = dst_t[0];
    
    /*printf("%f, %f %f, %f %f, %f %f, %f\n", dt, m1, m2, l1, l2, I1, I2, dst);*/
    
    int numIts = (int)round(dt/dst);
/*
    bool flgOnlyAccs = false;
    if(dt-1e-3 < 1.0 && dt+1e-3 > 1.0)
        flgOnlyAccs = true;
    
    if(flgOnlyAccs)
        plhs[0] = mxCreateDoubleMatrix(2, 1, mxREAL); /* MODE 2 */
 /*   else
        plhs[0] = mxCreateDoubleMatrix(6, 1, mxREAL); /* MODE 1 */
   //
    //double *outArray;
    //outArray = mxGetPr(plhs[0]);
    
   /* double a1 = xValues[0];
    a1 += M_PI;
    double a1d = xValues[1];
    double a2 = xValues[2]; 
    double a2d = xValues[3];*/
    
    double s1, c1, s2, c2;
  	double h11, h12, h21, h22, b1, b2;
  	double determinant;
  	double a1dd, a2dd;

	double l1CM = l1 / 2.0;
	double l2CM = l2 / 2.0;
    
    double tmpTau1, tmpTau2, t1, t2;
    double tau1, tau2;
    double a1, a1d, a2, a2d;

    int sampleIndex, rowOffset, outOffset;
    for(sampleIndex = 0; sampleIndex < numSamples; sampleIndex++){
        rowOffset = sampleIndex * 2;
        tau1 = u_t[0 + rowOffset];
        tau2 = u_t[1 + rowOffset];

        rowOffset = sampleIndex * 4;
        a1  = xValues[0 + rowOffset] + M_PI;
        a1d = xValues[1 + rowOffset];
        a2  = xValues[2 + rowOffset];
        a2d = xValues[3 + rowOffset];

        int tIndex;
        for(tIndex=0;tIndex<numIts;tIndex++)
        {

            s1 = sin( a1 );
            c1 = cos( a1 );
            s2 = sin( a2 );
            c2 = cos( a2 );

            /*printf("a1 %f s1 %f c1 %f\n",a1 , s1, c1);
            printf("a2 %f s2 %f c2 %f\n",a2 , s2, c2);*/
        
            h11 = I1 + I2  + l1CM * l1CM * m1 + l1 * l1 * m2 + l2CM * l2CM * m2 + 2 * l1 * l2CM * m2 * c2;
            h12 = I2 + l2CM * l2CM * m2 + l1 * l2CM * m2 * c2;

            b1 = g * l1CM * m1 * s1 + g * l1 * m2 * s1 + g * l2CM * m2 * c2 * s1 - 2 * a1d * a2d * l1 * l2CM * m2 * s2 - a2d * a2d * l1 * l2CM * m2 * s2 + g *l2CM * m2 * c1 * s2;

            h21 = I2 + l2CM * l2CM * m2 + l1 * l2CM * m2 * c2;
            h22 = I2 + l2CM * l2CM * m2;

            b2 = g * l2CM * m2 * c2 * s1 + a1d * a1d * l1 * l2CM * m2 * s2 + g * l2CM * m2 * c1 * s2;

        
            /*printf("h11 %f h12 %f b1 %f\n",h11 , h12, b1);
            printf("h21 %f h22 %fb2 %f\n",h11 , h12, b2);*/
        
        
            /*
            * A = [h11 h12; h21 h22];
            * b = [tau1 - b1 ; tau2 - b2 ];
            * acceleration = A ^-1 * b; %Ainv = [h22 -h12 ; -h21 h11]/determinant;
            */
        
        
            tmpTau1 = tau1;
            tmpTau2 = tau2;
        
            if(use_gains/*(mxGetM(prhs[9]) > 0) && (mxGetN(prhs[10]) > 0)*/){
                if( (a1 < -PDSetPoints[0])){
                    tmpTau1 += PDGains[0]*(-PDSetPoints[0]-a1) + PDGains[1]*(-PDSetPoints[1]-a1d);
                }
                if( (a1 > PDSetPoints[0])){
                    tmpTau1 += PDGains[0]*(PDSetPoints[0]-a1) + PDGains[1]*(PDSetPoints[1]-a1d);
                }
                if( (a2 < -PDSetPoints[2])){
                    tmpTau2 += PDGains[2]*(-PDSetPoints[2]-a2) + PDGains[3]*(-PDSetPoints[3]-a2d);
                }
                if( (a2 > PDSetPoints[2])){
                    tmpTau2 += PDGains[2]*(PDSetPoints[2]-a2) + PDGains[3]*(PDSetPoints[3]-a2d);
                }
                
           /* printf("PD 1 tau :%f a: %f %f sp: %f %f gains %f %f => tmptau: %f\n",tau1, a1, a1d, PDSetPoints[0], PDSetPoints[1], PDGains[0], PDGains[1], tmpTau1);
            printf("PD 2 tau :%f a: %f %f sp: %f %f gains %f %f => tmptau: %f\n",tau2, a2, a2d, PDSetPoints[2], PDSetPoints[3], PDGains[2], PDGains[3], tmpTau2);*/
            }
        
        
            tmpTau1 -= VISCOUS_FRICTION1*a1d;
            tmpTau2 -= VISCOUS_FRICTION2*a2d;
        
            determinant = h11 * h22 - h12 * h21;
        
        
            /*printf("determinant %f \n",determinant);*/
        
            a1dd = (h22 * (tmpTau1 - b1) - h12 * (tmpTau2 - b2))/determinant;
            a2dd = (h11 * (tmpTau2 - b2) - h21 * (tmpTau1 - b1))/determinant;
        
            /*printf("a1dd %f a2dd %f\n",a1dd , a2dd);*/
        
        
            a1 += dst*a1d;
            a1d += dst*a1dd;
            a2 += dst*a2d;
            a2d += dst*a2dd;

            if(dim_out == 2){
                outOffset = sampleIndex * 2;
                outArray[outOffset + 0] = a1dd;
                outArray[outOffset + 1] = a2dd;
            }
            else{
                outOffset = sampleIndex * 6;
                outArray[outOffset + 0] = (a1 - M_PI);
                outArray[outOffset + 1] = a1d;
                outArray[outOffset + 2] = a2;
                outArray[outOffset + 3] = a2d;
                outArray[outOffset + 4] = t1;
                outArray[outOffset + 5] = t2;
            }
        }
    
    /*printf("after sim: x1: %f, x2 %f, x3: %f, x4 %f\n", Phi1, dPhi1, Phi2, dPhi2);*/
    }
    return;
}
