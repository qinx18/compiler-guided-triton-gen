#include <math.h>

#define W 192
#define H 128

float imgIn[W][H];
float imgOut[W][H];
float yy1[W][H];
float y2[W][H];

float alpha = 0.25f;

// Local variables from kernel function
float xm1, tm1, ym1, ym2;
float xp1, xp2;
float tp1, tp2;
float yp1, yp2;
float k;
float a1, a2, a3, a4, a5, a6, a7, a8;
float b1, b2, c1, c2;

void deriche_kernel() {
  int i, j;
#pragma scop
   k = (1.0-expf(-alpha))*(1.0-expf(-alpha))/(1.0+2.0*alpha*expf(-alpha)-expf(2.0*alpha));
   a1 = a5 = k;
   a2 = a6 = k*expf(-alpha)*(alpha-1.0);
   a3 = a7 = k*expf(-alpha)*(alpha+1.0);
   a4 = a8 = -k*expf(-2.0*alpha);
   b1 =  powf(2.0,-alpha);
   b2 = -expf(-2.0*alpha);
   c1 = c2 = 1;

   for (i=0; i<W; i++) {
        ym1 = 0.0;
        ym2 = 0.0;
        xm1 = 0.0;
        for (j=0; j<H; j++) {
            yy1[i][j] = a1*imgIn[i][j] + a2*xm1 + b1*ym1 + b2*ym2;
            xm1 = imgIn[i][j];
            ym2 = ym1;
            ym1 = yy1[i][j];
        }
    }

    for (i=0; i<W; i++) {
        yp1 = 0.0;
        yp2 = 0.0;
        xp1 = 0.0;
        xp2 = 0.0;
        for (j=H-1; j>=0; j--) {
            y2[i][j] = a3*xp1 + a4*xp2 + b1*yp1 + b2*yp2;
            xp2 = xp1;
            xp1 = imgIn[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

    for (i=0; i<W; i++)
        for (j=0; j<H; j++) {
            imgOut[i][j] = c1 * (yy1[i][j] + y2[i][j]);
        }

    for (j=0; j<H; j++) {
        tm1 = 0.0;
        ym1 = 0.0;
        ym2 = 0.0;
        for (i=0; i<W; i++) {
            yy1[i][j] = a5*imgOut[i][j] + a6*tm1 + b1*ym1 + b2*ym2;
            tm1 = imgOut[i][j];
            ym2 = ym1;
            ym1 = yy1 [i][j];
        }
    }


    for (j=0; j<H; j++) {
        tp1 = 0.0;
        tp2 = 0.0;
        yp1 = 0.0;
        yp2 = 0.0;
        for (i=W-1; i>=0; i--) {
            y2[i][j] = a7*tp1 + a8*tp2 + b1*yp1 + b2*yp2;
            tp2 = tp1;
            tp1 = imgOut[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

    for (i=0; i<W; i++)
        for (j=0; j<H; j++)
            imgOut[i][j] = c2*(yy1[i][j] + y2[i][j]);
#pragma endscop
}
