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
/* Generated on Tue May 18 13:55:12 EDT 1999 */

#include <fftw-int.h>
#include <fftw.h>

/* Generated by: ./genfft -magic-alignment-check -magic-twiddle-load-all -magic-variables 4 -magic-loopi -hc2real 32 */

/*
 * This function contains 156 FP additions, 54 FP multiplications,
 * (or, 140 additions, 38 multiplications, 16 fused multiply/add),
 * 44 stack variables, and 64 memory accesses
 */
static const fftw_real K765366864 = FFTW_KONST(+0.765366864730179543456919968060797733522689125);
static const fftw_real K1_847759065 = FFTW_KONST(+1.847759065022573512256366378793576573644833252);
static const fftw_real K555570233 = FFTW_KONST(+0.555570233019602224742830813948532874374937191);
static const fftw_real K831469612 = FFTW_KONST(+0.831469612302545237078788377617905756738560812);
static const fftw_real K195090322 = FFTW_KONST(+0.195090322016128267848284868477022240927691618);
static const fftw_real K980785280 = FFTW_KONST(+0.980785280403230449126182236134239036973933731);
static const fftw_real K1_414213562 = FFTW_KONST(+1.414213562373095048801688724209698078569671875);
static const fftw_real K2_000000000 = FFTW_KONST(+2.000000000000000000000000000000000000000000000);

/*
 * Generator Id's : 
 * $Id$
 * $Id$
 * $Id$
 */

void fftw_hc2real_32(const fftw_real *real_input, const fftw_real *imag_input, fftw_real *output, int real_istride, int imag_istride, int ostride)
{
     fftw_real tmp9;
     fftw_real tmp134;
     fftw_real tmp96;
     fftw_real tmp37;
     fftw_real tmp32;
     fftw_real tmp58;
     fftw_real tmp56;
     fftw_real tmp80;
     fftw_real tmp145;
     fftw_real tmp149;
     fftw_real tmp119;
     fftw_real tmp123;
     fftw_real tmp6;
     fftw_real tmp34;
     fftw_real tmp93;
     fftw_real tmp133;
     fftw_real tmp17;
     fftw_real tmp39;
     fftw_real tmp46;
     fftw_real tmp83;
     fftw_real tmp100;
     fftw_real tmp136;
     fftw_real tmp103;
     fftw_real tmp137;
     fftw_real tmp25;
     fftw_real tmp49;
     fftw_real tmp65;
     fftw_real tmp79;
     fftw_real tmp142;
     fftw_real tmp148;
     fftw_real tmp112;
     fftw_real tmp122;
     ASSERT_ALIGNED_DOUBLE();
     {
	  fftw_real tmp7;
	  fftw_real tmp8;
	  fftw_real tmp94;
	  fftw_real tmp35;
	  fftw_real tmp36;
	  fftw_real tmp95;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp7 = real_input[4 * real_istride];
	  tmp8 = real_input[12 * real_istride];
	  tmp94 = tmp7 - tmp8;
	  tmp35 = imag_input[4 * imag_istride];
	  tmp36 = imag_input[12 * imag_istride];
	  tmp95 = tmp36 + tmp35;
	  tmp9 = K2_000000000 * (tmp7 + tmp8);
	  tmp134 = K1_414213562 * (tmp94 + tmp95);
	  tmp96 = K1_414213562 * (tmp94 - tmp95);
	  tmp37 = K2_000000000 * (tmp35 - tmp36);
     }
     {
	  fftw_real tmp28;
	  fftw_real tmp113;
	  fftw_real tmp55;
	  fftw_real tmp117;
	  fftw_real tmp31;
	  fftw_real tmp116;
	  fftw_real tmp52;
	  fftw_real tmp114;
	  ASSERT_ALIGNED_DOUBLE();
	  {
	       fftw_real tmp26;
	       fftw_real tmp27;
	       fftw_real tmp53;
	       fftw_real tmp54;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp26 = real_input[3 * real_istride];
	       tmp27 = real_input[13 * real_istride];
	       tmp28 = tmp26 + tmp27;
	       tmp113 = tmp26 - tmp27;
	       tmp53 = imag_input[3 * imag_istride];
	       tmp54 = imag_input[13 * imag_istride];
	       tmp55 = tmp53 - tmp54;
	       tmp117 = tmp53 + tmp54;
	  }
	  {
	       fftw_real tmp29;
	       fftw_real tmp30;
	       fftw_real tmp50;
	       fftw_real tmp51;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp29 = real_input[5 * real_istride];
	       tmp30 = real_input[11 * real_istride];
	       tmp31 = tmp29 + tmp30;
	       tmp116 = tmp30 - tmp29;
	       tmp50 = imag_input[5 * imag_istride];
	       tmp51 = imag_input[11 * imag_istride];
	       tmp52 = tmp50 - tmp51;
	       tmp114 = tmp51 + tmp50;
	  }
	  tmp32 = tmp28 + tmp31;
	  tmp58 = tmp31 - tmp28;
	  tmp56 = tmp52 + tmp55;
	  tmp80 = tmp55 - tmp52;
	  {
	       fftw_real tmp143;
	       fftw_real tmp144;
	       fftw_real tmp115;
	       fftw_real tmp118;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp143 = tmp113 + tmp114;
	       tmp144 = tmp117 - tmp116;
	       tmp145 = (K980785280 * tmp143) - (K195090322 * tmp144);
	       tmp149 = (K195090322 * tmp143) + (K980785280 * tmp144);
	       tmp115 = tmp113 - tmp114;
	       tmp118 = tmp116 + tmp117;
	       tmp119 = (K831469612 * tmp115) - (K555570233 * tmp118);
	       tmp123 = (K555570233 * tmp115) + (K831469612 * tmp118);
	  }
     }
     {
	  fftw_real tmp5;
	  fftw_real tmp92;
	  fftw_real tmp3;
	  fftw_real tmp90;
	  ASSERT_ALIGNED_DOUBLE();
	  {
	       fftw_real tmp4;
	       fftw_real tmp91;
	       fftw_real tmp1;
	       fftw_real tmp2;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp4 = real_input[8 * real_istride];
	       tmp5 = K2_000000000 * tmp4;
	       tmp91 = imag_input[8 * imag_istride];
	       tmp92 = K2_000000000 * tmp91;
	       tmp1 = real_input[0];
	       tmp2 = real_input[16 * real_istride];
	       tmp3 = tmp1 + tmp2;
	       tmp90 = tmp1 - tmp2;
	  }
	  tmp6 = tmp3 + tmp5;
	  tmp34 = tmp3 - tmp5;
	  tmp93 = tmp90 - tmp92;
	  tmp133 = tmp90 + tmp92;
     }
     {
	  fftw_real tmp13;
	  fftw_real tmp98;
	  fftw_real tmp45;
	  fftw_real tmp102;
	  fftw_real tmp16;
	  fftw_real tmp101;
	  fftw_real tmp42;
	  fftw_real tmp99;
	  ASSERT_ALIGNED_DOUBLE();
	  {
	       fftw_real tmp11;
	       fftw_real tmp12;
	       fftw_real tmp43;
	       fftw_real tmp44;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp11 = real_input[2 * real_istride];
	       tmp12 = real_input[14 * real_istride];
	       tmp13 = tmp11 + tmp12;
	       tmp98 = tmp11 - tmp12;
	       tmp43 = imag_input[2 * imag_istride];
	       tmp44 = imag_input[14 * imag_istride];
	       tmp45 = tmp43 - tmp44;
	       tmp102 = tmp43 + tmp44;
	  }
	  {
	       fftw_real tmp14;
	       fftw_real tmp15;
	       fftw_real tmp40;
	       fftw_real tmp41;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp14 = real_input[6 * real_istride];
	       tmp15 = real_input[10 * real_istride];
	       tmp16 = tmp14 + tmp15;
	       tmp101 = tmp15 - tmp14;
	       tmp40 = imag_input[6 * imag_istride];
	       tmp41 = imag_input[10 * imag_istride];
	       tmp42 = tmp40 - tmp41;
	       tmp99 = tmp41 + tmp40;
	  }
	  tmp17 = K2_000000000 * (tmp13 + tmp16);
	  tmp39 = tmp13 - tmp16;
	  tmp46 = tmp42 + tmp45;
	  tmp83 = K2_000000000 * (tmp45 - tmp42);
	  tmp100 = tmp98 - tmp99;
	  tmp136 = tmp98 + tmp99;
	  tmp103 = tmp101 + tmp102;
	  tmp137 = tmp102 - tmp101;
     }
     {
	  fftw_real tmp21;
	  fftw_real tmp106;
	  fftw_real tmp64;
	  fftw_real tmp110;
	  fftw_real tmp24;
	  fftw_real tmp109;
	  fftw_real tmp61;
	  fftw_real tmp107;
	  ASSERT_ALIGNED_DOUBLE();
	  {
	       fftw_real tmp19;
	       fftw_real tmp20;
	       fftw_real tmp62;
	       fftw_real tmp63;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp19 = real_input[real_istride];
	       tmp20 = real_input[15 * real_istride];
	       tmp21 = tmp19 + tmp20;
	       tmp106 = tmp19 - tmp20;
	       tmp62 = imag_input[imag_istride];
	       tmp63 = imag_input[15 * imag_istride];
	       tmp64 = tmp62 - tmp63;
	       tmp110 = tmp62 + tmp63;
	  }
	  {
	       fftw_real tmp22;
	       fftw_real tmp23;
	       fftw_real tmp59;
	       fftw_real tmp60;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp22 = real_input[7 * real_istride];
	       tmp23 = real_input[9 * real_istride];
	       tmp24 = tmp22 + tmp23;
	       tmp109 = tmp23 - tmp22;
	       tmp59 = imag_input[7 * imag_istride];
	       tmp60 = imag_input[9 * imag_istride];
	       tmp61 = tmp59 - tmp60;
	       tmp107 = tmp60 + tmp59;
	  }
	  tmp25 = tmp21 + tmp24;
	  tmp49 = tmp21 - tmp24;
	  tmp65 = tmp61 + tmp64;
	  tmp79 = tmp64 - tmp61;
	  {
	       fftw_real tmp140;
	       fftw_real tmp141;
	       fftw_real tmp108;
	       fftw_real tmp111;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp140 = tmp106 + tmp107;
	       tmp141 = tmp110 - tmp109;
	       tmp142 = (K555570233 * tmp140) + (K831469612 * tmp141);
	       tmp148 = (K831469612 * tmp140) - (K555570233 * tmp141);
	       tmp108 = tmp106 - tmp107;
	       tmp111 = tmp109 + tmp110;
	       tmp112 = (K980785280 * tmp108) - (K195090322 * tmp111);
	       tmp122 = (K195090322 * tmp108) + (K980785280 * tmp111);
	  }
     }
     {
	  fftw_real tmp33;
	  fftw_real tmp81;
	  fftw_real tmp18;
	  fftw_real tmp78;
	  fftw_real tmp10;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp33 = K2_000000000 * (tmp25 + tmp32);
	  tmp81 = K2_000000000 * (tmp79 - tmp80);
	  tmp10 = tmp6 + tmp9;
	  tmp18 = tmp10 + tmp17;
	  tmp78 = tmp10 - tmp17;
	  output[16 * ostride] = tmp18 - tmp33;
	  output[0] = tmp18 + tmp33;
	  output[8 * ostride] = tmp78 - tmp81;
	  output[24 * ostride] = tmp78 + tmp81;
     }
     {
	  fftw_real tmp84;
	  fftw_real tmp88;
	  fftw_real tmp87;
	  fftw_real tmp89;
	  fftw_real tmp82;
	  fftw_real tmp85;
	  fftw_real tmp86;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp82 = tmp6 - tmp9;
	  tmp84 = tmp82 - tmp83;
	  tmp88 = tmp82 + tmp83;
	  tmp85 = tmp25 - tmp32;
	  tmp86 = tmp79 + tmp80;
	  tmp87 = K1_414213562 * (tmp85 - tmp86);
	  tmp89 = K1_414213562 * (tmp85 + tmp86);
	  output[20 * ostride] = tmp84 - tmp87;
	  output[4 * ostride] = tmp84 + tmp87;
	  output[12 * ostride] = tmp88 - tmp89;
	  output[28 * ostride] = tmp88 + tmp89;
     }
     {
	  fftw_real tmp48;
	  fftw_real tmp68;
	  fftw_real tmp67;
	  fftw_real tmp69;
	  ASSERT_ALIGNED_DOUBLE();
	  {
	       fftw_real tmp38;
	       fftw_real tmp47;
	       fftw_real tmp57;
	       fftw_real tmp66;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp38 = tmp34 - tmp37;
	       tmp47 = K1_414213562 * (tmp39 - tmp46);
	       tmp48 = tmp38 + tmp47;
	       tmp68 = tmp38 - tmp47;
	       tmp57 = tmp49 - tmp56;
	       tmp66 = tmp58 + tmp65;
	       tmp67 = (K1_847759065 * tmp57) - (K765366864 * tmp66);
	       tmp69 = (K1_847759065 * tmp66) + (K765366864 * tmp57);
	  }
	  output[18 * ostride] = tmp48 - tmp67;
	  output[2 * ostride] = tmp48 + tmp67;
	  output[10 * ostride] = tmp68 - tmp69;
	  output[26 * ostride] = tmp68 + tmp69;
     }
     {
	  fftw_real tmp72;
	  fftw_real tmp76;
	  fftw_real tmp75;
	  fftw_real tmp77;
	  ASSERT_ALIGNED_DOUBLE();
	  {
	       fftw_real tmp70;
	       fftw_real tmp71;
	       fftw_real tmp73;
	       fftw_real tmp74;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp70 = tmp34 + tmp37;
	       tmp71 = K1_414213562 * (tmp39 + tmp46);
	       tmp72 = tmp70 - tmp71;
	       tmp76 = tmp70 + tmp71;
	       tmp73 = tmp49 + tmp56;
	       tmp74 = tmp65 - tmp58;
	       tmp75 = (K765366864 * tmp73) - (K1_847759065 * tmp74);
	       tmp77 = (K765366864 * tmp74) + (K1_847759065 * tmp73);
	  }
	  output[22 * ostride] = tmp72 - tmp75;
	  output[6 * ostride] = tmp72 + tmp75;
	  output[14 * ostride] = tmp76 - tmp77;
	  output[30 * ostride] = tmp76 + tmp77;
     }
     {
	  fftw_real tmp120;
	  fftw_real tmp124;
	  fftw_real tmp105;
	  fftw_real tmp121;
	  fftw_real tmp97;
	  fftw_real tmp104;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp120 = K2_000000000 * (tmp112 + tmp119);
	  tmp124 = K2_000000000 * (tmp122 - tmp123);
	  tmp97 = tmp93 + tmp96;
	  tmp104 = (K1_847759065 * tmp100) - (K765366864 * tmp103);
	  tmp105 = tmp97 + tmp104;
	  tmp121 = tmp97 - tmp104;
	  output[17 * ostride] = tmp105 - tmp120;
	  output[ostride] = tmp105 + tmp120;
	  output[9 * ostride] = tmp121 - tmp124;
	  output[25 * ostride] = tmp121 + tmp124;
     }
     {
	  fftw_real tmp127;
	  fftw_real tmp131;
	  fftw_real tmp130;
	  fftw_real tmp132;
	  ASSERT_ALIGNED_DOUBLE();
	  {
	       fftw_real tmp125;
	       fftw_real tmp126;
	       fftw_real tmp128;
	       fftw_real tmp129;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp125 = tmp93 - tmp96;
	       tmp126 = (K765366864 * tmp100) + (K1_847759065 * tmp103);
	       tmp127 = tmp125 - tmp126;
	       tmp131 = tmp125 + tmp126;
	       tmp128 = tmp112 - tmp119;
	       tmp129 = tmp122 + tmp123;
	       tmp130 = K1_414213562 * (tmp128 - tmp129);
	       tmp132 = K1_414213562 * (tmp128 + tmp129);
	  }
	  output[21 * ostride] = tmp127 - tmp130;
	  output[5 * ostride] = tmp127 + tmp130;
	  output[13 * ostride] = tmp131 - tmp132;
	  output[29 * ostride] = tmp131 + tmp132;
     }
     {
	  fftw_real tmp146;
	  fftw_real tmp150;
	  fftw_real tmp139;
	  fftw_real tmp147;
	  fftw_real tmp135;
	  fftw_real tmp138;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp146 = K2_000000000 * (tmp142 - tmp145);
	  tmp150 = K2_000000000 * (tmp148 - tmp149);
	  tmp135 = tmp133 - tmp134;
	  tmp138 = (K765366864 * tmp136) - (K1_847759065 * tmp137);
	  tmp139 = tmp135 - tmp138;
	  tmp147 = tmp135 + tmp138;
	  output[11 * ostride] = tmp139 - tmp146;
	  output[27 * ostride] = tmp139 + tmp146;
	  output[19 * ostride] = tmp147 - tmp150;
	  output[3 * ostride] = tmp147 + tmp150;
     }
     {
	  fftw_real tmp153;
	  fftw_real tmp157;
	  fftw_real tmp156;
	  fftw_real tmp158;
	  ASSERT_ALIGNED_DOUBLE();
	  {
	       fftw_real tmp151;
	       fftw_real tmp152;
	       fftw_real tmp154;
	       fftw_real tmp155;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp151 = tmp133 + tmp134;
	       tmp152 = (K1_847759065 * tmp136) + (K765366864 * tmp137);
	       tmp153 = tmp151 - tmp152;
	       tmp157 = tmp151 + tmp152;
	       tmp154 = tmp148 + tmp149;
	       tmp155 = tmp142 + tmp145;
	       tmp156 = K1_414213562 * (tmp154 - tmp155);
	       tmp158 = K1_414213562 * (tmp155 + tmp154);
	  }
	  output[23 * ostride] = tmp153 - tmp156;
	  output[7 * ostride] = tmp153 + tmp156;
	  output[15 * ostride] = tmp157 - tmp158;
	  output[31 * ostride] = tmp157 + tmp158;
     }
}

fftw_codelet_desc fftw_hc2real_32_desc =
{
     "fftw_hc2real_32",
     (void (*)()) fftw_hc2real_32,
     32,
     FFTW_BACKWARD,
     FFTW_HC2REAL,
     719,
     0,
     (const int *) 0,
};
