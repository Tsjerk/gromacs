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
/* Generated on Tue May 18 13:54:33 EDT 1999 */

#include <fftw-int.h>
#include <fftw.h>

/* Generated by: ./genfft -magic-alignment-check -magic-twiddle-load-all -magic-variables 4 -magic-loopi -real2hc 3 */

/*
 * This function contains 4 FP additions, 2 FP multiplications,
 * (or, 3 additions, 1 multiplications, 1 fused multiply/add),
 * 4 stack variables, and 6 memory accesses
 */
static const fftw_real K866025403 = FFTW_KONST(+0.866025403784438646763723170752936183471402627);
static const fftw_real K500000000 = FFTW_KONST(+0.500000000000000000000000000000000000000000000);

/*
 * Generator Id's : 
 * $Id$
 * $Id$
 * $Id$
 */

void fftw_real2hc_3(const fftw_real *input, fftw_real *real_output, fftw_real *imag_output, int istride, int real_ostride, int imag_ostride)
{
     fftw_real tmp1;
     fftw_real tmp2;
     fftw_real tmp3;
     fftw_real tmp4;
     ASSERT_ALIGNED_DOUBLE();
     tmp1 = input[0];
     tmp2 = input[istride];
     tmp3 = input[2 * istride];
     tmp4 = tmp2 + tmp3;
     real_output[real_ostride] = tmp1 - (K500000000 * tmp4);
     real_output[0] = tmp1 + tmp4;
     imag_output[imag_ostride] = K866025403 * (tmp3 - tmp2);
}

fftw_codelet_desc fftw_real2hc_3_desc =
{
     "fftw_real2hc_3",
     (void (*)()) fftw_real2hc_3,
     3,
     FFTW_FORWARD,
     FFTW_REAL2HC,
     68,
     0,
     (const int *) 0,
};
