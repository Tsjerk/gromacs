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
/* Generated on Tue May 18 13:54:27 EDT 1999 */

#include <fftw-int.h>
#include <fftw.h>

/* Generated by: ./genfft -magic-alignment-check -magic-twiddle-load-all -magic-variables 4 -magic-loopi -notwiddle 6 */

/*
 * This function contains 36 FP additions, 8 FP multiplications,
 * (or, 32 additions, 4 multiplications, 4 fused multiply/add),
 * 20 stack variables, and 24 memory accesses
 */
static const fftw_real K866025403 = FFTW_KONST(+0.866025403784438646763723170752936183471402627);
static const fftw_real K500000000 = FFTW_KONST(+0.500000000000000000000000000000000000000000000);

/*
 * Generator Id's : 
 * $Id$
 * $Id$
 * $Id$
 */

void fftw_no_twiddle_6(const fftw_complex *input, fftw_complex *output, int istride, int ostride)
{
     fftw_real tmp3;
     fftw_real tmp11;
     fftw_real tmp26;
     fftw_real tmp33;
     fftw_real tmp6;
     fftw_real tmp12;
     fftw_real tmp9;
     fftw_real tmp13;
     fftw_real tmp10;
     fftw_real tmp14;
     fftw_real tmp18;
     fftw_real tmp30;
     fftw_real tmp21;
     fftw_real tmp31;
     fftw_real tmp27;
     fftw_real tmp34;
     ASSERT_ALIGNED_DOUBLE();
     {
	  fftw_real tmp1;
	  fftw_real tmp2;
	  fftw_real tmp24;
	  fftw_real tmp25;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp1 = c_re(input[0]);
	  tmp2 = c_re(input[3 * istride]);
	  tmp3 = tmp1 - tmp2;
	  tmp11 = tmp1 + tmp2;
	  tmp24 = c_im(input[0]);
	  tmp25 = c_im(input[3 * istride]);
	  tmp26 = tmp24 - tmp25;
	  tmp33 = tmp24 + tmp25;
     }
     {
	  fftw_real tmp4;
	  fftw_real tmp5;
	  fftw_real tmp7;
	  fftw_real tmp8;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp4 = c_re(input[2 * istride]);
	  tmp5 = c_re(input[5 * istride]);
	  tmp6 = tmp4 - tmp5;
	  tmp12 = tmp4 + tmp5;
	  tmp7 = c_re(input[4 * istride]);
	  tmp8 = c_re(input[istride]);
	  tmp9 = tmp7 - tmp8;
	  tmp13 = tmp7 + tmp8;
     }
     tmp10 = tmp6 + tmp9;
     tmp14 = tmp12 + tmp13;
     {
	  fftw_real tmp16;
	  fftw_real tmp17;
	  fftw_real tmp19;
	  fftw_real tmp20;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp16 = c_im(input[2 * istride]);
	  tmp17 = c_im(input[5 * istride]);
	  tmp18 = tmp16 - tmp17;
	  tmp30 = tmp16 + tmp17;
	  tmp19 = c_im(input[4 * istride]);
	  tmp20 = c_im(input[istride]);
	  tmp21 = tmp19 - tmp20;
	  tmp31 = tmp19 + tmp20;
     }
     tmp27 = tmp18 + tmp21;
     tmp34 = tmp30 + tmp31;
     {
	  fftw_real tmp15;
	  fftw_real tmp22;
	  fftw_real tmp29;
	  fftw_real tmp32;
	  ASSERT_ALIGNED_DOUBLE();
	  c_re(output[3 * ostride]) = tmp3 + tmp10;
	  tmp15 = tmp3 - (K500000000 * tmp10);
	  tmp22 = K866025403 * (tmp18 - tmp21);
	  c_re(output[5 * ostride]) = tmp15 - tmp22;
	  c_re(output[ostride]) = tmp15 + tmp22;
	  c_re(output[0]) = tmp11 + tmp14;
	  tmp29 = tmp11 - (K500000000 * tmp14);
	  tmp32 = K866025403 * (tmp30 - tmp31);
	  c_re(output[2 * ostride]) = tmp29 - tmp32;
	  c_re(output[4 * ostride]) = tmp29 + tmp32;
     }
     {
	  fftw_real tmp23;
	  fftw_real tmp28;
	  fftw_real tmp35;
	  fftw_real tmp36;
	  ASSERT_ALIGNED_DOUBLE();
	  c_im(output[3 * ostride]) = tmp26 + tmp27;
	  tmp23 = K866025403 * (tmp9 - tmp6);
	  tmp28 = tmp26 - (K500000000 * tmp27);
	  c_im(output[ostride]) = tmp23 + tmp28;
	  c_im(output[5 * ostride]) = tmp28 - tmp23;
	  c_im(output[0]) = tmp33 + tmp34;
	  tmp35 = tmp33 - (K500000000 * tmp34);
	  tmp36 = K866025403 * (tmp13 - tmp12);
	  c_im(output[2 * ostride]) = tmp35 - tmp36;
	  c_im(output[4 * ostride]) = tmp36 + tmp35;
     }
}

fftw_codelet_desc fftw_no_twiddle_6_desc =
{
     "fftw_no_twiddle_6",
     (void (*)()) fftw_no_twiddle_6,
     6,
     FFTW_FORWARD,
     FFTW_NOTW,
     133,
     0,
     (const int *) 0,
};
