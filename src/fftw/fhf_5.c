/*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

/* This file was automatically generated --- DO NOT EDIT */
/* Generated on Tue May 18 13:55:30 EDT 1999 */

#include <fftw-int.h>
#include <fftw.h>

/* Generated by: ./genfft -magic-alignment-check -magic-twiddle-load-all -magic-variables 4 -magic-loopi -hc2hc-forward 5 */

/*
 * This function contains 64 FP additions, 40 FP multiplications,
 * (or, 44 additions, 20 multiplications, 20 fused multiply/add),
 * 27 stack variables, and 40 memory accesses
 */
static const fftw_real K250000000 = FFTW_KONST(+0.250000000000000000000000000000000000000000000);
static const fftw_real K559016994 = FFTW_KONST(+0.559016994374947424102293417182819058860154590);
static const fftw_real K587785252 = FFTW_KONST(+0.587785252292473129168705954639072768597652438);
static const fftw_real K951056516 = FFTW_KONST(+0.951056516295153572116439333379382143405698634);

/*
 * Generator Id's : 
 * $Id$
 * $Id$
 * $Id$
 */

void fftw_hc2hc_forward_5(fftw_real *A, const fftw_complex *W, int iostride, int m, int dist)
{
     int i;
     fftw_real *X;
     fftw_real *Y;
     X = A;
     Y = A + (5 * iostride);
     {
	  fftw_real tmp70;
	  fftw_real tmp67;
	  fftw_real tmp68;
	  fftw_real tmp63;
	  fftw_real tmp71;
	  fftw_real tmp66;
	  fftw_real tmp69;
	  fftw_real tmp72;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp70 = X[0];
	  {
	       fftw_real tmp61;
	       fftw_real tmp62;
	       fftw_real tmp64;
	       fftw_real tmp65;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp61 = X[4 * iostride];
	       tmp62 = X[iostride];
	       tmp67 = tmp62 + tmp61;
	       tmp64 = X[2 * iostride];
	       tmp65 = X[3 * iostride];
	       tmp68 = tmp64 + tmp65;
	       tmp63 = tmp61 - tmp62;
	       tmp71 = tmp67 + tmp68;
	       tmp66 = tmp64 - tmp65;
	  }
	  Y[-iostride] = (K951056516 * tmp63) - (K587785252 * tmp66);
	  Y[-2 * iostride] = (K587785252 * tmp63) + (K951056516 * tmp66);
	  X[0] = tmp70 + tmp71;
	  tmp69 = K559016994 * (tmp67 - tmp68);
	  tmp72 = tmp70 - (K250000000 * tmp71);
	  X[iostride] = tmp69 + tmp72;
	  X[2 * iostride] = tmp72 - tmp69;
     }
     X = X + dist;
     Y = Y - dist;
     for (i = 2; i < m; i = i + 2, X = X + dist, Y = Y - dist, W = W + 4) {
	  fftw_real tmp13;
	  fftw_real tmp52;
	  fftw_real tmp42;
	  fftw_real tmp45;
	  fftw_real tmp49;
	  fftw_real tmp50;
	  fftw_real tmp51;
	  fftw_real tmp54;
	  fftw_real tmp53;
	  fftw_real tmp24;
	  fftw_real tmp35;
	  fftw_real tmp36;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp13 = X[0];
	  tmp52 = Y[-4 * iostride];
	  {
	       fftw_real tmp18;
	       fftw_real tmp40;
	       fftw_real tmp34;
	       fftw_real tmp44;
	       fftw_real tmp23;
	       fftw_real tmp41;
	       fftw_real tmp29;
	       fftw_real tmp43;
	       ASSERT_ALIGNED_DOUBLE();
	       {
		    fftw_real tmp15;
		    fftw_real tmp17;
		    fftw_real tmp14;
		    fftw_real tmp16;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp15 = X[iostride];
		    tmp17 = Y[-3 * iostride];
		    tmp14 = c_re(W[0]);
		    tmp16 = c_im(W[0]);
		    tmp18 = (tmp14 * tmp15) - (tmp16 * tmp17);
		    tmp40 = (tmp16 * tmp15) + (tmp14 * tmp17);
	       }
	       {
		    fftw_real tmp31;
		    fftw_real tmp33;
		    fftw_real tmp30;
		    fftw_real tmp32;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp31 = X[3 * iostride];
		    tmp33 = Y[-iostride];
		    tmp30 = c_re(W[2]);
		    tmp32 = c_im(W[2]);
		    tmp34 = (tmp30 * tmp31) - (tmp32 * tmp33);
		    tmp44 = (tmp32 * tmp31) + (tmp30 * tmp33);
	       }
	       {
		    fftw_real tmp20;
		    fftw_real tmp22;
		    fftw_real tmp19;
		    fftw_real tmp21;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp20 = X[4 * iostride];
		    tmp22 = Y[0];
		    tmp19 = c_re(W[3]);
		    tmp21 = c_im(W[3]);
		    tmp23 = (tmp19 * tmp20) - (tmp21 * tmp22);
		    tmp41 = (tmp21 * tmp20) + (tmp19 * tmp22);
	       }
	       {
		    fftw_real tmp26;
		    fftw_real tmp28;
		    fftw_real tmp25;
		    fftw_real tmp27;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp26 = X[2 * iostride];
		    tmp28 = Y[-2 * iostride];
		    tmp25 = c_re(W[1]);
		    tmp27 = c_im(W[1]);
		    tmp29 = (tmp25 * tmp26) - (tmp27 * tmp28);
		    tmp43 = (tmp27 * tmp26) + (tmp25 * tmp28);
	       }
	       tmp42 = tmp40 - tmp41;
	       tmp45 = tmp43 - tmp44;
	       tmp49 = tmp40 + tmp41;
	       tmp50 = tmp43 + tmp44;
	       tmp51 = tmp49 + tmp50;
	       tmp54 = tmp29 - tmp34;
	       tmp53 = tmp18 - tmp23;
	       tmp24 = tmp18 + tmp23;
	       tmp35 = tmp29 + tmp34;
	       tmp36 = tmp24 + tmp35;
	  }
	  X[0] = tmp13 + tmp36;
	  {
	       fftw_real tmp46;
	       fftw_real tmp48;
	       fftw_real tmp39;
	       fftw_real tmp47;
	       fftw_real tmp37;
	       fftw_real tmp38;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp46 = (K951056516 * tmp42) + (K587785252 * tmp45);
	       tmp48 = (K951056516 * tmp45) - (K587785252 * tmp42);
	       tmp37 = K559016994 * (tmp24 - tmp35);
	       tmp38 = tmp13 - (K250000000 * tmp36);
	       tmp39 = tmp37 + tmp38;
	       tmp47 = tmp38 - tmp37;
	       Y[-4 * iostride] = tmp39 - tmp46;
	       X[iostride] = tmp39 + tmp46;
	       X[2 * iostride] = tmp47 - tmp48;
	       Y[-3 * iostride] = tmp47 + tmp48;
	  }
	  Y[0] = tmp51 + tmp52;
	  {
	       fftw_real tmp55;
	       fftw_real tmp60;
	       fftw_real tmp58;
	       fftw_real tmp59;
	       fftw_real tmp56;
	       fftw_real tmp57;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp55 = (K951056516 * tmp53) + (K587785252 * tmp54);
	       tmp60 = (K951056516 * tmp54) - (K587785252 * tmp53);
	       tmp56 = K559016994 * (tmp49 - tmp50);
	       tmp57 = tmp52 - (K250000000 * tmp51);
	       tmp58 = tmp56 + tmp57;
	       tmp59 = tmp57 - tmp56;
	       X[4 * iostride] = -(tmp55 + tmp58);
	       Y[-iostride] = tmp58 - tmp55;
	       X[3 * iostride] = -(tmp59 - tmp60);
	       Y[-2 * iostride] = tmp60 + tmp59;
	  }
     }
     if (i == m) {
	  fftw_real tmp8;
	  fftw_real tmp3;
	  fftw_real tmp6;
	  fftw_real tmp9;
	  fftw_real tmp12;
	  fftw_real tmp11;
	  fftw_real tmp7;
	  fftw_real tmp10;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp8 = X[0];
	  {
	       fftw_real tmp1;
	       fftw_real tmp2;
	       fftw_real tmp4;
	       fftw_real tmp5;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp1 = X[2 * iostride];
	       tmp2 = X[3 * iostride];
	       tmp3 = tmp1 - tmp2;
	       tmp4 = X[4 * iostride];
	       tmp5 = X[iostride];
	       tmp6 = tmp4 - tmp5;
	       tmp9 = tmp3 + tmp6;
	       tmp12 = tmp4 + tmp5;
	       tmp11 = tmp1 + tmp2;
	  }
	  X[2 * iostride] = tmp8 + tmp9;
	  tmp7 = K559016994 * (tmp3 - tmp6);
	  tmp10 = tmp8 - (K250000000 * tmp9);
	  X[0] = tmp7 + tmp10;
	  X[iostride] = tmp10 - tmp7;
	  Y[0] = -((K951056516 * tmp11) + (K587785252 * tmp12));
	  Y[-iostride] = -((K951056516 * tmp12) - (K587785252 * tmp11));
     }
}

static const int twiddle_order[] =
{1, 2, 3, 4};
fftw_codelet_desc fftw_hc2hc_forward_5_desc =
{
     "fftw_hc2hc_forward_5",
     (void (*)()) fftw_hc2hc_forward_5,
     5,
     FFTW_FORWARD,
     FFTW_HC2HC,
     113,
     4,
     twiddle_order,
};
