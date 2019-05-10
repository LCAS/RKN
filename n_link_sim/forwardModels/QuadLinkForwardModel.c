#include "math.h"
#include "stdio.h"
#include "omp.h"
#define SQ(x) ((x)*(x))
/*
 * logic deciding which mode to use now in python since swig only supports using arrays whose size is know before
 * the function is called:

 * quad pendulum forward model
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
              double G,
              double *k_t, int dim_k,
              double dst,
              double *outArray, int num_out, int dim_out){
    double tau1, tau2, tau3, tau4;

    double m1 = m_t[0];
    double m2 = m_t[1];
    double m3 = m_t[2];
    double m4 = m_t[3];
    double l1 = l_t[0];
    double l2 = l_t[1];
    double l3 = l_t[2];
    double l4 = l_t[3];
    double I1 = I_t[0];
    double I2 = I_t[1];
    double I3 = I_t[2];
    double I4 = I_t[3];


    double VISCOUS_FRICTION1 = k_t[0];
    double VISCOUS_FRICTION2 = k_t[1];
    double VISCOUS_FRICTION3 = k_t[2];
    double VISCOUS_FRICTION4 = k_t[3];

   /* printf("%f, %f %f %f %f, %f %f %f %f, %f %f %f %f\n", dt, m1, m2, m3, m4, l1, l2, l3, l4, I1, I2, I3, I4); */

    int numIts = (int)round(dt/dst);

    double a1, a1d, a2 , a2d, a3, a3d, a4, a4d;

    /* This code is taken from Chris Atkeson's page (http://www.cs.cmu.edu/~cga/kdc/dynamics-2d/dynamics4.c), all credit belongs to him...*/

    double s1, c1, s2, c2, s3, c3, s4, c4;
    double a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t;
    double determinant;
    double s12, c12, s23, c23, s34, c34, s1234, s123, s234, c234;
    double a1d_a1d, a2d_a2d, a3d_a3d, a4d_a4d;
    double a1d_p_a2d_2, l4cm_m4, l3_m4, l3cm_m3, l2cm_m2, l3cm_m3_l3_m4;
    double l2cm_m2_p_l2_m3_p_m4;
    double l3_l4cm_m4, l2_l4cm_m4, l1_l4cm_m4;
    double l2_l3cm_m3_l3_m4, l1_l3cm_m3_l3_m4, l2_l4cm_m4_c34;
    double expr1, expr2, expr3, expr4, expr5, expr6, expr7, expr8;
    double expr4a, expr4b, expr5a, expr9a, expr9;
    double a123d, l1_l3cm_m3_l3_m4_s23, l2_l4cm_m4_s34;
    double a1dd, a2dd, a3dd, a4dd;

    double l1cm = l1 / 2.0;
    double l2cm = l2 / 2.0;
    double l3cm = l3 / 2.0;
    double l4cm = l4 / 2.0;

    double t1, t2, t3, t4;
   // printf("T: %d \n", omp_get_max_threads());

    int sampleIndex, rowOffset, outOffset;
    #pragma omp parallel for private(a1, a1d, a2 , a2d, a3, a3d, a4, a4d,\
                                     s1, c1, s2, c2, s3, c3, s4, c4,\
                                     a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t,\
                                     determinant,\
                                     s12, c12, s23, c23, s34, c34, s1234, s123, s234, c234,\
                                     a1d_a1d, a2d_a2d, a3d_a3d, a4d_a4d,\
                                     a1d_p_a2d_2, l4cm_m4, l3_m4, l3cm_m3, l2cm_m2, l3cm_m3_l3_m4,\
                                     l2cm_m2_p_l2_m3_p_m4,\
                                     l3_l4cm_m4, l2_l4cm_m4, l1_l4cm_m4,\
                                     l2_l3cm_m3_l3_m4, l1_l3cm_m3_l3_m4, l2_l4cm_m4_c34,\
                                     expr1, expr2, expr3, expr4, expr5, expr6, expr7, expr8,\
                                     expr4a, expr4b, expr5a, expr9a, expr9,\
                                     a123d, l1_l3cm_m3_l3_m4_s23, l2_l4cm_m4_s34,\
                                     a1dd, a2dd, a3dd, a4dd,\
                                     t1, t2, t3, t4,\
                                     rowOffset, outOffset)
    for (sampleIndex=0;sampleIndex<numSamples;sampleIndex++)
    {
        rowOffset=sampleIndex*4;
        tau1 = u_t[0 + rowOffset];
        tau2 = u_t[1 + rowOffset];
        tau3 = u_t[2 + rowOffset];
        tau4 = u_t[3 + rowOffset];
        rowOffset=sampleIndex*8;
        a1 = xValues[0 + rowOffset] + M_PI;
        a1d = xValues[1 + rowOffset];
        a2 = xValues[2 + rowOffset];
        a2d = xValues[3 + rowOffset];
        a3 = xValues[4 + rowOffset];
        a3d = xValues[5 + rowOffset];
        a4 = xValues[6 + rowOffset];
        a4d = xValues[7 + rowOffset];
   /* printf("line 114, %i, %f, %f,%f,%f\n", numSamples, tau1, tau2, a1,a1d);       getchar(); */


        int tIndex;
        for(tIndex=0;tIndex<numIts;tIndex++)
        {

            s1 = sin( a1 );
            c1 = cos( a1 );
            s2 = sin( a2 );
            c2 = cos( a2 );
            s3 = sin( a3 );
            c3 = cos( a3 );
            s4 = sin( a4 );
            c4 = cos( a4 );
            s12 = s1*c2 + c1*s2;
            c12 = c1*c2 - s1*s2;
            s23 = s2*c3 + c2*s3;
            c23 = c2*c3 - s2*s3;
            s34 = s3*c4 + c3*s4;
            c34 = c3*c4 - s3*s4;
            s1234 = s12*c34 + c12*s34;
            s123 = s12*c3 + c12*s3;
            s234 = s2*c34 + c2*s34;
            c234 = c2*c34 - s2*s34;

            a1d_a1d = a1d*a1d;
            a2d_a2d = a2d*a2d;
            a3d_a3d = a3d*a3d;
            a4d_a4d = a4d*a4d;
            a1d_p_a2d_2 = (a1d + a2d)*(a1d + a2d);

            l4cm_m4 = l4cm*m4;
            l3_l4cm_m4 = l3*l4cm_m4;
            l2_l4cm_m4 = l2*l4cm_m4;
            l2_l4cm_m4_c34 = l2_l4cm_m4*c34;
            l1_l4cm_m4 = l1*l4cm_m4;
            l3_m4 = l3*m4;
            l3cm_m3 = l3cm*m3;
            l3cm_m3_l3_m4 = l3cm_m3 + l3_m4;
            l2cm_m2 = l2cm*m2;
            l2cm_m2_p_l2_m3_p_m4 = l2cm_m2 + l2*(m3 + m4);
            l2_l3cm_m3_l3_m4 = l2*l3cm_m3_l3_m4;
            l1_l3cm_m3_l3_m4 = l1*l3cm_m3_l3_m4;
            a123d = a1d + a2d + a3d;
            l1_l3cm_m3_l3_m4_s23 = l1_l3cm_m3_l3_m4*s23;
            l2_l4cm_m4_s34 = l2_l4cm_m4*s34;

            expr1 = G*(s123*l3cm_m3_l3_m4 + s1234*l4cm_m4);
            expr2 = (2*a123d + a4d)*a4d*l3_l4cm_m4*s4;
            expr3 = G*l2cm_m2_p_l2_m3_p_m4*s12;
            expr4a = 2*a1d*a4d + 2*a2d*a4d + 2*a3d*a4d + a4d_a4d;
            expr4b = 2*a1d*a3d + 2*a2d*a3d + a3d_a3d;
            expr4 = (expr4b + expr4a)*l2_l4cm_m4_s34;
            expr5a = a1d_a1d*l1*s234;
            expr5 = l4cm_m4*expr5a;
            expr6 = expr4b*l2_l3cm_m3_l3_m4*s3;
            expr7 = l1*l2cm_m2_p_l2_m3_p_m4;
            expr8 = l1*(m2+m3+m4);
            expr9a = 2*a1d*a2d + a2d_a2d;
            expr9 = (expr9a + expr4b);

            /* Fourth row */
            p = I4 + l4cm*l4cm_m4;

            o = p + l3_l4cm_m4*c4;

            n = o + l2_l4cm_m4_c34;

            m = n + l1_l4cm_m4*c234;

            t4 = - VISCOUS_FRICTION4*a4d
                    -(l4cm_m4*(a123d*a123d*l3*s4 +
                    a1d_p_a2d_2*l2*s34 +
                    expr5a + G*s1234));

            t = tau4 - VISCOUS_FRICTION4*a4d
                    -(l4cm_m4*(a123d*a123d*l3*s4 +
                    a1d_p_a2d_2*l2*s34 +
                    expr5a + G*s1234));

            /* Third row */
            l = o;

            k = I3 + o + l3cm*l3cm_m3 + l3*l3_m4 + l3_l4cm_m4*c4;

            j = k + l2_l3cm_m3_l3_m4*c3 + l2_l4cm_m4_c34;

            i = j + l1_l3cm_m3_l3_m4*c23
                    + l1_l4cm_m4*c234;

            t3 = - VISCOUS_FRICTION3*a3d
                    -((a1d_p_a2d_2*l2_l3cm_m3_l3_m4*s3 + a1d_a1d*l1_l3cm_m3_l3_m4_s23) +
                    - expr2
                    + a1d_p_a2d_2*l2_l4cm_m4_s34
                    + expr5
                    + expr1
                    );

            s = tau3 - VISCOUS_FRICTION3*a3d
                    -((a1d_p_a2d_2*l2_l3cm_m3_l3_m4*s3 + a1d_a1d*l1_l3cm_m3_l3_m4_s23) +
                    - expr2
                    + a1d_p_a2d_2*l2_l4cm_m4_s34
                    + expr5
                    + expr1
                    );

            /* Second row */
            h = n;

            g = j;

            f = j + I2 + l2cm*l2cm_m2  + SQ(l2)*(m3 + m4)
            + l2_l3cm_m3_l3_m4*c3 + l2_l4cm_m4_c34;

            e = f + i - j + expr7*c2;

            t2 = - VISCOUS_FRICTION2*a2d
                    - (
                    a1d_a1d*expr7*s2
                    - expr6
                    + a1d_a1d*l1_l3cm_m3_l3_m4_s23
                    - expr2
                    - expr4
                    + expr5
                    + expr3
                    + expr1
                    );

            r = tau2 - VISCOUS_FRICTION2*a2d
                    - (
                    a1d_a1d*expr7*s2
                    - expr6
                    + a1d_a1d*l1_l3cm_m3_l3_m4_s23
                    - expr2
                    - expr4
                    + expr5
                    + expr3
                    + expr1
                    );

            /* First row */
            d = m;

            c = i;

            b = e;

            a = 2*e + I1 - f + SQ(l1cm)*m1 + l1*expr8;

            t1 = - VISCOUS_FRICTION1*a1d
                    - ( -expr9a*expr7*s2
                    - expr6
                    - expr9*l1_l3cm_m3_l3_m4_s23
                    - expr2
                    - expr4
                    - (expr9 + expr4a)*l1_l4cm_m4*s234
                    + expr3
                    + G*(l1cm*m1 + expr8)*s1
                    + expr1
                    );

            q = tau1 - VISCOUS_FRICTION1*a1d
                    - ( -expr9a*expr7*s2
                    - expr6
                    - expr9*l1_l3cm_m3_l3_m4_s23
                    - expr2
                    - expr4
                    - (expr9 + expr4a)*l1_l4cm_m4*s234
                    + expr3
                    + G*(l1cm*m1 + expr8)*s1
                    + expr1
                    );

            determinant =
                    (d*g*j*m - c*h*j*m - d*f*k*m + b*h*k*m + c*f*l*m - b*g*l*m - d*g*i*n +
                    c*h*i*n + d*e*k*n - a*h*k*n - c*e*l*n + a*g*l*n + d*f*i*o - b*h*i*o -
                    d*e*j*o + a*h*j*o + b*e*l*o - a*f*l*o - c*f*i*p + b*g*i*p + c*e*j*p -
                    a*g*j*p - b*e*k*p + a*f*k*p);

            a1dd = q*(-(h*k*n) + g*l*n + h*j*o - f*l*o - g*j*p + f*k*p)
            + r*(d*k*n - c*l*n - d*j*o + b*l*o + c*j*p - b*k*p)
            + s*(-(d*g*n) + c*h*n + d*f*o - b*h*o - c*f*p + b*g*p)
            + t*(d*g*j - c*h*j - d*f*k + b*h*k + c*f*l - b*g*l);

            a2dd = q*(h*k*m - g*l*m - h*i*o + e*l*o + g*i*p - e*k*p)
            + r*(-(d*k*m) + c*l*m + d*i*o - a*l*o - c*i*p + a*k*p)
            + s*(d*g*m - c*h*m - d*e*o + a*h*o + c*e*p - a*g*p)
            + t*(-(d*g*i) + c*h*i + d*e*k - a*h*k - c*e*l + a*g*l);

            a3dd = q*(-(h*j*m) + f*l*m + h*i*n - e*l*n - f*i*p + e*j*p)
            + r*(d*j*m - b*l*m - d*i*n + a*l*n + b*i*p - a*j*p)
            + s*(-(d*f*m) + b*h*m + d*e*n - a*h*n - b*e*p + a*f*p)
            + t*(d*f*i - b*h*i - d*e*j + a*h*j + b*e*l - a*f*l);

            a4dd = q*(g*j*m - f*k*m - g*i*n + e*k*n + f*i*o - e*j*o)
            + r*(-(c*j*m) + b*k*m + c*i*n - a*k*n - b*i*o + a*j*o)
            + s*(c*f*m - b*g*m - c*e*n + a*g*n + b*e*o - a*f*o)
            + t*(-(c*f*i) + b*g*i + c*e*j - a*g*j - b*e*k + a*f*k);
            a1dd = a1dd/determinant;
            a2dd = a2dd/determinant;
            a3dd = a3dd/determinant;
            a4dd = a4dd/determinant;


            a1 += dst*a1d;
            a1d += dst*a1dd;
            a2 += dst*a2d;
            a2d += dst*a2dd;
            a3 += dst*a3d;
            a3d += dst*a3dd;
            a4 += dst*a4d;
            a4d += dst*a4dd;
        }

        /*printf("after sim: x1: %f, x2 %f, x3: %f, x4 %f\n", Phi1, dPhi1, Phi2, dPhi2);*/

        if(dim_out == 4)
        {
            outOffset=sampleIndex*4;
            outArray[outOffset+0] = a1dd;
            outArray[outOffset+1] = a2dd;
            outArray[outOffset+2] = a3dd;
            outArray[outOffset+3] = a4dd;
        }
        else
        {
            outOffset=sampleIndex*12;
            outArray[outOffset] = (a1 - M_PI);
            outArray[outOffset+1] = a1d;
            outArray[outOffset+2] = a2;
            outArray[outOffset+3] = a2d;
            outArray[outOffset+4] = a3;
            outArray[outOffset+5] = a3d;
            outArray[outOffset+6] = a4;
            outArray[outOffset+7] = a4d;
            outArray[outOffset+8] = t1;
            outArray[outOffset+9] = t2;
            outArray[outOffset+10] = t3;
            outArray[outOffset+11] = t4;
        }
    }
    return;
}