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
/* Generated on Tue May 18 13:54:55 EDT 1999 */

#include <fftw-int.h>
#include <fftw.h>

/* Generated by: ./genfft -magic-alignment-check -magic-twiddle-load-all -magic-variables 4 -magic-loopi -notwiddleinv 15 */

/*
 * This function contains 156 FP additions, 56 FP multiplications,
 * (or, 128 additions, 28 multiplications, 28 fused multiply/add),
 * 62 stack variables, and 60 memory accesses
 */
static const fftw_real K951056516 = FFTW_KONST(+0.951056516295153572116439333379382143405698634);
static const fftw_real K587785252 = FFTW_KONST(+0.587785252292473129168705954639072768597652438);
static const fftw_real K250000000 = FFTW_KONST(+0.250000000000000000000000000000000000000000000);
static const fftw_real K559016994 = FFTW_KONST(+0.559016994374947424102293417182819058860154590);
static const fftw_real K500000000 = FFTW_KONST(+0.500000000000000000000000000000000000000000000);
static const fftw_real K866025403 = FFTW_KONST(+0.866025403784438646763723170752936183471402627);

/*
 * Generator Id's : 
 * $Id$
 * $Id$
 * $Id$
 */

void fftwi_no_twiddle_15(const fftw_complex *input, fftw_complex *output, int istride, int ostride)
{
     fftw_real tmp5;
     fftw_real tmp121;
     fftw_real tmp148;
     fftw_real tmp87;
     fftw_real tmp35;
     fftw_real tmp67;
     fftw_real tmp21;
     fftw_real tmp26;
     fftw_real tmp27;
     fftw_real tmp111;
     fftw_real tmp114;
     fftw_real tmp123;
     fftw_real tmp139;
     fftw_real tmp140;
     fftw_real tmp146;
     fftw_real tmp81;
     fftw_real tmp82;
     fftw_real tmp89;
     fftw_real tmp71;
     fftw_real tmp72;
     fftw_real tmp73;
     fftw_real tmp57;
     fftw_real tmp64;
     fftw_real tmp65;
     fftw_real tmp10;
     fftw_real tmp15;
     fftw_real tmp16;
     fftw_real tmp104;
     fftw_real tmp107;
     fftw_real tmp122;
     fftw_real tmp136;
     fftw_real tmp137;
     fftw_real tmp145;
     fftw_real tmp78;
     fftw_real tmp79;
     fftw_real tmp88;
     fftw_real tmp68;
     fftw_real tmp69;
     fftw_real tmp70;
     fftw_real tmp42;
     fftw_real tmp49;
     fftw_real tmp50;
     ASSERT_ALIGNED_DOUBLE();
     {
	  fftw_real tmp1;
	  fftw_real tmp30;
	  fftw_real tmp4;
	  fftw_real tmp29;
	  fftw_real tmp33;
	  fftw_real tmp120;
	  fftw_real tmp119;
	  fftw_real tmp34;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp1 = c_re(input[0]);
	  tmp30 = c_im(input[0]);
	  {
	       fftw_real tmp2;
	       fftw_real tmp3;
	       fftw_real tmp31;
	       fftw_real tmp32;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp2 = c_re(input[5 * istride]);
	       tmp3 = c_re(input[10 * istride]);
	       tmp4 = tmp2 + tmp3;
	       tmp29 = K866025403 * (tmp2 - tmp3);
	       tmp31 = c_im(input[5 * istride]);
	       tmp32 = c_im(input[10 * istride]);
	       tmp33 = tmp31 + tmp32;
	       tmp120 = K866025403 * (tmp32 - tmp31);
	  }
	  tmp5 = tmp1 + tmp4;
	  tmp119 = tmp1 - (K500000000 * tmp4);
	  tmp121 = tmp119 - tmp120;
	  tmp148 = tmp119 + tmp120;
	  tmp87 = tmp30 + tmp33;
	  tmp34 = tmp30 - (K500000000 * tmp33);
	  tmp35 = tmp29 + tmp34;
	  tmp67 = tmp34 - tmp29;
     }
     {
	  fftw_real tmp17;
	  fftw_real tmp20;
	  fftw_real tmp51;
	  fftw_real tmp109;
	  fftw_real tmp52;
	  fftw_real tmp55;
	  fftw_real tmp56;
	  fftw_real tmp110;
	  fftw_real tmp22;
	  fftw_real tmp25;
	  fftw_real tmp58;
	  fftw_real tmp112;
	  fftw_real tmp59;
	  fftw_real tmp62;
	  fftw_real tmp63;
	  fftw_real tmp113;
	  ASSERT_ALIGNED_DOUBLE();
	  {
	       fftw_real tmp18;
	       fftw_real tmp19;
	       fftw_real tmp53;
	       fftw_real tmp54;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp17 = c_re(input[6 * istride]);
	       tmp18 = c_re(input[11 * istride]);
	       tmp19 = c_re(input[istride]);
	       tmp20 = tmp18 + tmp19;
	       tmp51 = K866025403 * (tmp18 - tmp19);
	       tmp109 = tmp17 - (K500000000 * tmp20);
	       tmp52 = c_im(input[6 * istride]);
	       tmp53 = c_im(input[11 * istride]);
	       tmp54 = c_im(input[istride]);
	       tmp55 = tmp53 + tmp54;
	       tmp56 = tmp52 - (K500000000 * tmp55);
	       tmp110 = K866025403 * (tmp54 - tmp53);
	  }
	  {
	       fftw_real tmp23;
	       fftw_real tmp24;
	       fftw_real tmp60;
	       fftw_real tmp61;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp22 = c_re(input[9 * istride]);
	       tmp23 = c_re(input[14 * istride]);
	       tmp24 = c_re(input[4 * istride]);
	       tmp25 = tmp23 + tmp24;
	       tmp58 = K866025403 * (tmp23 - tmp24);
	       tmp112 = tmp22 - (K500000000 * tmp25);
	       tmp59 = c_im(input[9 * istride]);
	       tmp60 = c_im(input[14 * istride]);
	       tmp61 = c_im(input[4 * istride]);
	       tmp62 = tmp60 + tmp61;
	       tmp63 = tmp59 - (K500000000 * tmp62);
	       tmp113 = K866025403 * (tmp61 - tmp60);
	  }
	  tmp21 = tmp17 + tmp20;
	  tmp26 = tmp22 + tmp25;
	  tmp27 = tmp21 + tmp26;
	  tmp111 = tmp109 - tmp110;
	  tmp114 = tmp112 - tmp113;
	  tmp123 = tmp111 + tmp114;
	  tmp139 = tmp109 + tmp110;
	  tmp140 = tmp112 + tmp113;
	  tmp146 = tmp139 + tmp140;
	  tmp81 = tmp52 + tmp55;
	  tmp82 = tmp59 + tmp62;
	  tmp89 = tmp81 + tmp82;
	  tmp71 = tmp56 - tmp51;
	  tmp72 = tmp63 - tmp58;
	  tmp73 = tmp71 + tmp72;
	  tmp57 = tmp51 + tmp56;
	  tmp64 = tmp58 + tmp63;
	  tmp65 = tmp57 + tmp64;
     }
     {
	  fftw_real tmp6;
	  fftw_real tmp9;
	  fftw_real tmp36;
	  fftw_real tmp102;
	  fftw_real tmp37;
	  fftw_real tmp40;
	  fftw_real tmp41;
	  fftw_real tmp103;
	  fftw_real tmp11;
	  fftw_real tmp14;
	  fftw_real tmp43;
	  fftw_real tmp105;
	  fftw_real tmp44;
	  fftw_real tmp47;
	  fftw_real tmp48;
	  fftw_real tmp106;
	  ASSERT_ALIGNED_DOUBLE();
	  {
	       fftw_real tmp7;
	       fftw_real tmp8;
	       fftw_real tmp38;
	       fftw_real tmp39;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp6 = c_re(input[3 * istride]);
	       tmp7 = c_re(input[8 * istride]);
	       tmp8 = c_re(input[13 * istride]);
	       tmp9 = tmp7 + tmp8;
	       tmp36 = K866025403 * (tmp7 - tmp8);
	       tmp102 = tmp6 - (K500000000 * tmp9);
	       tmp37 = c_im(input[3 * istride]);
	       tmp38 = c_im(input[8 * istride]);
	       tmp39 = c_im(input[13 * istride]);
	       tmp40 = tmp38 + tmp39;
	       tmp41 = tmp37 - (K500000000 * tmp40);
	       tmp103 = K866025403 * (tmp39 - tmp38);
	  }
	  {
	       fftw_real tmp12;
	       fftw_real tmp13;
	       fftw_real tmp45;
	       fftw_real tmp46;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp11 = c_re(input[12 * istride]);
	       tmp12 = c_re(input[2 * istride]);
	       tmp13 = c_re(input[7 * istride]);
	       tmp14 = tmp12 + tmp13;
	       tmp43 = K866025403 * (tmp12 - tmp13);
	       tmp105 = tmp11 - (K500000000 * tmp14);
	       tmp44 = c_im(input[12 * istride]);
	       tmp45 = c_im(input[2 * istride]);
	       tmp46 = c_im(input[7 * istride]);
	       tmp47 = tmp45 + tmp46;
	       tmp48 = tmp44 - (K500000000 * tmp47);
	       tmp106 = K866025403 * (tmp46 - tmp45);
	  }
	  tmp10 = tmp6 + tmp9;
	  tmp15 = tmp11 + tmp14;
	  tmp16 = tmp10 + tmp15;
	  tmp104 = tmp102 - tmp103;
	  tmp107 = tmp105 - tmp106;
	  tmp122 = tmp104 + tmp107;
	  tmp136 = tmp102 + tmp103;
	  tmp137 = tmp105 + tmp106;
	  tmp145 = tmp136 + tmp137;
	  tmp78 = tmp37 + tmp40;
	  tmp79 = tmp44 + tmp47;
	  tmp88 = tmp78 + tmp79;
	  tmp68 = tmp41 - tmp36;
	  tmp69 = tmp48 - tmp43;
	  tmp70 = tmp68 + tmp69;
	  tmp42 = tmp36 + tmp41;
	  tmp49 = tmp43 + tmp48;
	  tmp50 = tmp42 + tmp49;
     }
     {
	  fftw_real tmp76;
	  fftw_real tmp28;
	  fftw_real tmp75;
	  fftw_real tmp84;
	  fftw_real tmp86;
	  fftw_real tmp80;
	  fftw_real tmp83;
	  fftw_real tmp85;
	  fftw_real tmp77;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp76 = K559016994 * (tmp16 - tmp27);
	  tmp28 = tmp16 + tmp27;
	  tmp75 = tmp5 - (K250000000 * tmp28);
	  tmp80 = tmp78 - tmp79;
	  tmp83 = tmp81 - tmp82;
	  tmp84 = (K587785252 * tmp80) - (K951056516 * tmp83);
	  tmp86 = (K951056516 * tmp80) + (K587785252 * tmp83);
	  c_re(output[0]) = tmp5 + tmp28;
	  tmp85 = tmp76 + tmp75;
	  c_re(output[6 * ostride]) = tmp85 - tmp86;
	  c_re(output[9 * ostride]) = tmp85 + tmp86;
	  tmp77 = tmp75 - tmp76;
	  c_re(output[12 * ostride]) = tmp77 - tmp84;
	  c_re(output[3 * ostride]) = tmp77 + tmp84;
     }
     {
	  fftw_real tmp134;
	  fftw_real tmp66;
	  fftw_real tmp133;
	  fftw_real tmp142;
	  fftw_real tmp144;
	  fftw_real tmp138;
	  fftw_real tmp141;
	  fftw_real tmp143;
	  fftw_real tmp135;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp134 = K559016994 * (tmp50 - tmp65);
	  tmp66 = tmp50 + tmp65;
	  tmp133 = tmp35 - (K250000000 * tmp66);
	  tmp138 = tmp136 - tmp137;
	  tmp141 = tmp139 - tmp140;
	  tmp142 = (K587785252 * tmp138) - (K951056516 * tmp141);
	  tmp144 = (K951056516 * tmp138) + (K587785252 * tmp141);
	  c_im(output[10 * ostride]) = tmp35 + tmp66;
	  tmp143 = tmp134 + tmp133;
	  c_im(output[4 * ostride]) = tmp143 - tmp144;
	  c_im(output[ostride]) = tmp143 + tmp144;
	  tmp135 = tmp133 - tmp134;
	  c_im(output[13 * ostride]) = tmp135 - tmp142;
	  c_im(output[7 * ostride]) = tmp135 + tmp142;
     }
     {
	  fftw_real tmp147;
	  fftw_real tmp149;
	  fftw_real tmp150;
	  fftw_real tmp154;
	  fftw_real tmp156;
	  fftw_real tmp152;
	  fftw_real tmp153;
	  fftw_real tmp155;
	  fftw_real tmp151;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp147 = K559016994 * (tmp145 - tmp146);
	  tmp149 = tmp145 + tmp146;
	  tmp150 = tmp148 - (K250000000 * tmp149);
	  tmp152 = tmp42 - tmp49;
	  tmp153 = tmp57 - tmp64;
	  tmp154 = (K951056516 * tmp152) + (K587785252 * tmp153);
	  tmp156 = (K587785252 * tmp152) - (K951056516 * tmp153);
	  c_re(output[10 * ostride]) = tmp148 + tmp149;
	  tmp155 = tmp150 - tmp147;
	  c_re(output[7 * ostride]) = tmp155 - tmp156;
	  c_re(output[13 * ostride]) = tmp156 + tmp155;
	  tmp151 = tmp147 + tmp150;
	  c_re(output[ostride]) = tmp151 - tmp154;
	  c_re(output[4 * ostride]) = tmp154 + tmp151;
     }
     {
	  fftw_real tmp126;
	  fftw_real tmp124;
	  fftw_real tmp125;
	  fftw_real tmp130;
	  fftw_real tmp132;
	  fftw_real tmp128;
	  fftw_real tmp129;
	  fftw_real tmp131;
	  fftw_real tmp127;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp126 = K559016994 * (tmp122 - tmp123);
	  tmp124 = tmp122 + tmp123;
	  tmp125 = tmp121 - (K250000000 * tmp124);
	  tmp128 = tmp68 - tmp69;
	  tmp129 = tmp71 - tmp72;
	  tmp130 = (K587785252 * tmp128) - (K951056516 * tmp129);
	  tmp132 = (K951056516 * tmp128) + (K587785252 * tmp129);
	  c_re(output[5 * ostride]) = tmp121 + tmp124;
	  tmp131 = tmp126 + tmp125;
	  c_re(output[11 * ostride]) = tmp131 - tmp132;
	  c_re(output[14 * ostride]) = tmp132 + tmp131;
	  tmp127 = tmp125 - tmp126;
	  c_re(output[2 * ostride]) = tmp127 - tmp130;
	  c_re(output[8 * ostride]) = tmp130 + tmp127;
     }
     {
	  fftw_real tmp92;
	  fftw_real tmp90;
	  fftw_real tmp91;
	  fftw_real tmp96;
	  fftw_real tmp97;
	  fftw_real tmp94;
	  fftw_real tmp95;
	  fftw_real tmp98;
	  fftw_real tmp93;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp92 = K559016994 * (tmp88 - tmp89);
	  tmp90 = tmp88 + tmp89;
	  tmp91 = tmp87 - (K250000000 * tmp90);
	  tmp94 = tmp10 - tmp15;
	  tmp95 = tmp21 - tmp26;
	  tmp96 = (K587785252 * tmp94) - (K951056516 * tmp95);
	  tmp97 = (K951056516 * tmp94) + (K587785252 * tmp95);
	  c_im(output[0]) = tmp87 + tmp90;
	  tmp98 = tmp92 + tmp91;
	  c_im(output[6 * ostride]) = tmp97 + tmp98;
	  c_im(output[9 * ostride]) = tmp98 - tmp97;
	  tmp93 = tmp91 - tmp92;
	  c_im(output[3 * ostride]) = tmp93 - tmp96;
	  c_im(output[12 * ostride]) = tmp96 + tmp93;
     }
     {
	  fftw_real tmp100;
	  fftw_real tmp74;
	  fftw_real tmp99;
	  fftw_real tmp116;
	  fftw_real tmp118;
	  fftw_real tmp108;
	  fftw_real tmp115;
	  fftw_real tmp117;
	  fftw_real tmp101;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp100 = K559016994 * (tmp70 - tmp73);
	  tmp74 = tmp70 + tmp73;
	  tmp99 = tmp67 - (K250000000 * tmp74);
	  tmp108 = tmp104 - tmp107;
	  tmp115 = tmp111 - tmp114;
	  tmp116 = (K587785252 * tmp108) - (K951056516 * tmp115);
	  tmp118 = (K951056516 * tmp108) + (K587785252 * tmp115);
	  c_im(output[5 * ostride]) = tmp67 + tmp74;
	  tmp117 = tmp100 + tmp99;
	  c_im(output[14 * ostride]) = tmp117 - tmp118;
	  c_im(output[11 * ostride]) = tmp117 + tmp118;
	  tmp101 = tmp99 - tmp100;
	  c_im(output[8 * ostride]) = tmp101 - tmp116;
	  c_im(output[2 * ostride]) = tmp101 + tmp116;
     }
}

fftw_codelet_desc fftwi_no_twiddle_15_desc =
{
     "fftwi_no_twiddle_15",
     (void (*)()) fftwi_no_twiddle_15,
     15,
     FFTW_BACKWARD,
     FFTW_NOTW,
     342,
     0,
     (const int *) 0,
};
