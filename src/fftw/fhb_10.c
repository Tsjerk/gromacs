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
/* Generated on Tue May 18 13:56:06 EDT 1999 */

#include <fftw-int.h>
#include <fftw.h>

/* Generated by: ./genfft -magic-alignment-check -magic-twiddle-load-all -magic-variables 4 -magic-loopi -hc2hc-backward 10 */

/*
 * This function contains 168 FP additions, 90 FP multiplications,
 * (or, 124 additions, 46 multiplications, 44 fused multiply/add),
 * 37 stack variables, and 80 memory accesses
 */
static const fftw_real K250000000 = FFTW_KONST(+0.250000000000000000000000000000000000000000000);
static const fftw_real K951056516 = FFTW_KONST(+0.951056516295153572116439333379382143405698634);
static const fftw_real K587785252 = FFTW_KONST(+0.587785252292473129168705954639072768597652438);
static const fftw_real K559016994 = FFTW_KONST(+0.559016994374947424102293417182819058860154590);
static const fftw_real K500000000 = FFTW_KONST(+0.500000000000000000000000000000000000000000000);
static const fftw_real K1_902113032 = FFTW_KONST(+1.902113032590307144232878666758764286811397268);
static const fftw_real K1_175570504 = FFTW_KONST(+1.175570504584946258337411909278145537195304875);
static const fftw_real K2_000000000 = FFTW_KONST(+2.000000000000000000000000000000000000000000000);
static const fftw_real K1_118033988 = FFTW_KONST(+1.118033988749894848204586834365638117720309180);

/*
 * Generator Id's : 
 * $Id$
 * $Id$
 * $Id$
 */

void fftw_hc2hc_backward_10(fftw_real *A, const fftw_complex *W, int iostride, int m, int dist)
{
     int i;
     fftw_real *X;
     fftw_real *Y;
     X = A;
     Y = A + (10 * iostride);
     {
	  fftw_real tmp155;
	  fftw_real tmp163;
	  fftw_real tmp175;
	  fftw_real tmp183;
	  fftw_real tmp172;
	  fftw_real tmp182;
	  fftw_real tmp162;
	  fftw_real tmp180;
	  fftw_real tmp166;
	  fftw_real tmp167;
	  fftw_real tmp170;
	  fftw_real tmp171;
	  ASSERT_ALIGNED_DOUBLE();
	  {
	       fftw_real tmp153;
	       fftw_real tmp154;
	       fftw_real tmp173;
	       fftw_real tmp174;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp153 = X[0];
	       tmp154 = X[5 * iostride];
	       tmp155 = tmp153 - tmp154;
	       tmp163 = tmp153 + tmp154;
	       tmp173 = Y[-4 * iostride];
	       tmp174 = Y[-iostride];
	       tmp175 = tmp173 - tmp174;
	       tmp183 = tmp173 + tmp174;
	  }
	  tmp170 = Y[-2 * iostride];
	  tmp171 = Y[-3 * iostride];
	  tmp172 = tmp170 - tmp171;
	  tmp182 = tmp170 + tmp171;
	  {
	       fftw_real tmp158;
	       fftw_real tmp164;
	       fftw_real tmp161;
	       fftw_real tmp165;
	       ASSERT_ALIGNED_DOUBLE();
	       {
		    fftw_real tmp156;
		    fftw_real tmp157;
		    fftw_real tmp159;
		    fftw_real tmp160;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp156 = X[2 * iostride];
		    tmp157 = X[3 * iostride];
		    tmp158 = tmp156 - tmp157;
		    tmp164 = tmp156 + tmp157;
		    tmp159 = X[4 * iostride];
		    tmp160 = X[iostride];
		    tmp161 = tmp159 - tmp160;
		    tmp165 = tmp159 + tmp160;
	       }
	       tmp162 = tmp158 + tmp161;
	       tmp180 = K1_118033988 * (tmp158 - tmp161);
	       tmp166 = tmp164 + tmp165;
	       tmp167 = K1_118033988 * (tmp164 - tmp165);
	  }
	  X[5 * iostride] = tmp155 + (K2_000000000 * tmp162);
	  {
	       fftw_real tmp184;
	       fftw_real tmp186;
	       fftw_real tmp181;
	       fftw_real tmp185;
	       fftw_real tmp179;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp184 = (K1_175570504 * tmp182) - (K1_902113032 * tmp183);
	       tmp186 = (K1_902113032 * tmp182) + (K1_175570504 * tmp183);
	       tmp179 = tmp155 - (K500000000 * tmp162);
	       tmp181 = tmp179 - tmp180;
	       tmp185 = tmp179 + tmp180;
	       X[7 * iostride] = tmp181 - tmp184;
	       X[3 * iostride] = tmp181 + tmp184;
	       X[iostride] = tmp185 - tmp186;
	       X[9 * iostride] = tmp185 + tmp186;
	  }
	  X[0] = tmp163 + (K2_000000000 * tmp166);
	  {
	       fftw_real tmp176;
	       fftw_real tmp178;
	       fftw_real tmp169;
	       fftw_real tmp177;
	       fftw_real tmp168;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp176 = (K1_902113032 * tmp172) + (K1_175570504 * tmp175);
	       tmp178 = (K1_902113032 * tmp175) - (K1_175570504 * tmp172);
	       tmp168 = tmp163 - (K500000000 * tmp166);
	       tmp169 = tmp167 + tmp168;
	       tmp177 = tmp168 - tmp167;
	       X[4 * iostride] = tmp169 + tmp176;
	       X[6 * iostride] = tmp169 - tmp176;
	       X[8 * iostride] = tmp177 - tmp178;
	       X[2 * iostride] = tmp177 + tmp178;
	  }
     }
     X = X + dist;
     Y = Y - dist;
     for (i = 2; i < m; i = i + 2, X = X + dist, Y = Y - dist, W = W + 9) {
	  fftw_real tmp35;
	  fftw_real tmp102;
	  fftw_real tmp77;
	  fftw_real tmp112;
	  fftw_real tmp72;
	  fftw_real tmp73;
	  fftw_real tmp50;
	  fftw_real tmp53;
	  fftw_real tmp123;
	  fftw_real tmp122;
	  fftw_real tmp109;
	  fftw_real tmp131;
	  fftw_real tmp61;
	  fftw_real tmp68;
	  fftw_real tmp80;
	  fftw_real tmp82;
	  fftw_real tmp134;
	  fftw_real tmp133;
	  fftw_real tmp119;
	  fftw_real tmp126;
	  ASSERT_ALIGNED_DOUBLE();
	  {
	       fftw_real tmp33;
	       fftw_real tmp34;
	       fftw_real tmp75;
	       fftw_real tmp76;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp33 = X[0];
	       tmp34 = Y[-5 * iostride];
	       tmp35 = tmp33 + tmp34;
	       tmp102 = tmp33 - tmp34;
	       tmp75 = Y[0];
	       tmp76 = X[5 * iostride];
	       tmp77 = tmp75 - tmp76;
	       tmp112 = tmp75 + tmp76;
	  }
	  {
	       fftw_real tmp38;
	       fftw_real tmp103;
	       fftw_real tmp48;
	       fftw_real tmp107;
	       fftw_real tmp41;
	       fftw_real tmp104;
	       fftw_real tmp45;
	       fftw_real tmp106;
	       ASSERT_ALIGNED_DOUBLE();
	       {
		    fftw_real tmp36;
		    fftw_real tmp37;
		    fftw_real tmp46;
		    fftw_real tmp47;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp36 = X[2 * iostride];
		    tmp37 = Y[-7 * iostride];
		    tmp38 = tmp36 + tmp37;
		    tmp103 = tmp36 - tmp37;
		    tmp46 = Y[-6 * iostride];
		    tmp47 = X[iostride];
		    tmp48 = tmp46 + tmp47;
		    tmp107 = tmp46 - tmp47;
	       }
	       {
		    fftw_real tmp39;
		    fftw_real tmp40;
		    fftw_real tmp43;
		    fftw_real tmp44;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp39 = Y[-8 * iostride];
		    tmp40 = X[3 * iostride];
		    tmp41 = tmp39 + tmp40;
		    tmp104 = tmp39 - tmp40;
		    tmp43 = X[4 * iostride];
		    tmp44 = Y[-9 * iostride];
		    tmp45 = tmp43 + tmp44;
		    tmp106 = tmp43 - tmp44;
	       }
	       {
		    fftw_real tmp42;
		    fftw_real tmp49;
		    fftw_real tmp105;
		    fftw_real tmp108;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp72 = tmp38 - tmp41;
		    tmp73 = tmp45 - tmp48;
		    tmp42 = tmp38 + tmp41;
		    tmp49 = tmp45 + tmp48;
		    tmp50 = tmp42 + tmp49;
		    tmp53 = K559016994 * (tmp42 - tmp49);
		    tmp123 = tmp106 - tmp107;
		    tmp122 = tmp103 - tmp104;
		    tmp105 = tmp103 + tmp104;
		    tmp108 = tmp106 + tmp107;
		    tmp109 = tmp105 + tmp108;
		    tmp131 = K559016994 * (tmp105 - tmp108);
	       }
	  }
	  {
	       fftw_real tmp57;
	       fftw_real tmp113;
	       fftw_real tmp67;
	       fftw_real tmp117;
	       fftw_real tmp60;
	       fftw_real tmp114;
	       fftw_real tmp64;
	       fftw_real tmp116;
	       ASSERT_ALIGNED_DOUBLE();
	       {
		    fftw_real tmp55;
		    fftw_real tmp56;
		    fftw_real tmp65;
		    fftw_real tmp66;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp55 = Y[-2 * iostride];
		    tmp56 = X[7 * iostride];
		    tmp57 = tmp55 - tmp56;
		    tmp113 = tmp55 + tmp56;
		    tmp65 = Y[-iostride];
		    tmp66 = X[6 * iostride];
		    tmp67 = tmp65 - tmp66;
		    tmp117 = tmp65 + tmp66;
	       }
	       {
		    fftw_real tmp58;
		    fftw_real tmp59;
		    fftw_real tmp62;
		    fftw_real tmp63;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp58 = Y[-3 * iostride];
		    tmp59 = X[8 * iostride];
		    tmp60 = tmp58 - tmp59;
		    tmp114 = tmp58 + tmp59;
		    tmp62 = Y[-4 * iostride];
		    tmp63 = X[9 * iostride];
		    tmp64 = tmp62 - tmp63;
		    tmp116 = tmp62 + tmp63;
	       }
	       {
		    fftw_real tmp78;
		    fftw_real tmp79;
		    fftw_real tmp115;
		    fftw_real tmp118;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp61 = tmp57 - tmp60;
		    tmp68 = tmp64 - tmp67;
		    tmp78 = tmp57 + tmp60;
		    tmp79 = tmp64 + tmp67;
		    tmp80 = tmp78 + tmp79;
		    tmp82 = K559016994 * (tmp78 - tmp79);
		    tmp134 = tmp116 + tmp117;
		    tmp133 = tmp113 + tmp114;
		    tmp115 = tmp113 - tmp114;
		    tmp118 = tmp116 - tmp117;
		    tmp119 = tmp115 + tmp118;
		    tmp126 = K559016994 * (tmp115 - tmp118);
	       }
	  }
	  X[0] = tmp35 + tmp50;
	  {
	       fftw_real tmp69;
	       fftw_real tmp91;
	       fftw_real tmp54;
	       fftw_real tmp90;
	       fftw_real tmp95;
	       fftw_real tmp74;
	       fftw_real tmp83;
	       fftw_real tmp94;
	       fftw_real tmp52;
	       fftw_real tmp81;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp69 = (K587785252 * tmp61) - (K951056516 * tmp68);
	       tmp91 = (K951056516 * tmp61) + (K587785252 * tmp68);
	       tmp52 = tmp35 - (K250000000 * tmp50);
	       tmp54 = tmp52 - tmp53;
	       tmp90 = tmp53 + tmp52;
	       tmp95 = (K951056516 * tmp72) + (K587785252 * tmp73);
	       tmp74 = (K587785252 * tmp72) - (K951056516 * tmp73);
	       tmp81 = tmp77 - (K250000000 * tmp80);
	       tmp83 = tmp81 - tmp82;
	       tmp94 = tmp82 + tmp81;
	       {
		    fftw_real tmp70;
		    fftw_real tmp84;
		    fftw_real tmp51;
		    fftw_real tmp71;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp70 = tmp54 - tmp69;
		    tmp84 = tmp74 + tmp83;
		    tmp51 = c_re(W[1]);
		    tmp71 = c_im(W[1]);
		    X[2 * iostride] = (tmp51 * tmp70) + (tmp71 * tmp84);
		    Y[-7 * iostride] = (tmp51 * tmp84) - (tmp71 * tmp70);
	       }
	       {
		    fftw_real tmp86;
		    fftw_real tmp88;
		    fftw_real tmp85;
		    fftw_real tmp87;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp86 = tmp54 + tmp69;
		    tmp88 = tmp83 - tmp74;
		    tmp85 = c_re(W[7]);
		    tmp87 = c_im(W[7]);
		    X[8 * iostride] = (tmp85 * tmp86) + (tmp87 * tmp88);
		    Y[-iostride] = (tmp85 * tmp88) - (tmp87 * tmp86);
	       }
	       {
		    fftw_real tmp92;
		    fftw_real tmp96;
		    fftw_real tmp89;
		    fftw_real tmp93;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp92 = tmp90 + tmp91;
		    tmp96 = tmp94 - tmp95;
		    tmp89 = c_re(W[3]);
		    tmp93 = c_im(W[3]);
		    X[4 * iostride] = (tmp89 * tmp92) + (tmp93 * tmp96);
		    Y[-5 * iostride] = (tmp89 * tmp96) - (tmp93 * tmp92);
	       }
	       {
		    fftw_real tmp98;
		    fftw_real tmp100;
		    fftw_real tmp97;
		    fftw_real tmp99;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp98 = tmp90 - tmp91;
		    tmp100 = tmp95 + tmp94;
		    tmp97 = c_re(W[5]);
		    tmp99 = c_im(W[5]);
		    X[6 * iostride] = (tmp97 * tmp98) + (tmp99 * tmp100);
		    Y[-3 * iostride] = (tmp97 * tmp100) - (tmp99 * tmp98);
	       }
	  }
	  Y[-9 * iostride] = tmp77 + tmp80;
	  {
	       fftw_real tmp110;
	       fftw_real tmp120;
	       fftw_real tmp101;
	       fftw_real tmp111;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp110 = tmp102 + tmp109;
	       tmp120 = tmp112 + tmp119;
	       tmp101 = c_re(W[4]);
	       tmp111 = c_im(W[4]);
	       X[5 * iostride] = (tmp101 * tmp110) + (tmp111 * tmp120);
	       Y[-4 * iostride] = (tmp101 * tmp120) - (tmp111 * tmp110);
	  }
	  {
	       fftw_real tmp124;
	       fftw_real tmp142;
	       fftw_real tmp127;
	       fftw_real tmp143;
	       fftw_real tmp147;
	       fftw_real tmp135;
	       fftw_real tmp132;
	       fftw_real tmp146;
	       fftw_real tmp125;
	       fftw_real tmp130;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp124 = (K587785252 * tmp122) - (K951056516 * tmp123);
	       tmp142 = (K951056516 * tmp122) + (K587785252 * tmp123);
	       tmp125 = tmp112 - (K250000000 * tmp119);
	       tmp127 = tmp125 - tmp126;
	       tmp143 = tmp126 + tmp125;
	       tmp147 = (K951056516 * tmp133) + (K587785252 * tmp134);
	       tmp135 = (K587785252 * tmp133) - (K951056516 * tmp134);
	       tmp130 = tmp102 - (K250000000 * tmp109);
	       tmp132 = tmp130 - tmp131;
	       tmp146 = tmp131 + tmp130;
	       {
		    fftw_real tmp128;
		    fftw_real tmp136;
		    fftw_real tmp121;
		    fftw_real tmp129;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp128 = tmp124 + tmp127;
		    tmp136 = tmp132 - tmp135;
		    tmp121 = c_re(W[6]);
		    tmp129 = c_im(W[6]);
		    Y[-2 * iostride] = (tmp121 * tmp128) - (tmp129 * tmp136);
		    X[7 * iostride] = (tmp129 * tmp128) + (tmp121 * tmp136);
	       }
	       {
		    fftw_real tmp138;
		    fftw_real tmp140;
		    fftw_real tmp137;
		    fftw_real tmp139;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp138 = tmp127 - tmp124;
		    tmp140 = tmp132 + tmp135;
		    tmp137 = c_re(W[2]);
		    tmp139 = c_im(W[2]);
		    Y[-6 * iostride] = (tmp137 * tmp138) - (tmp139 * tmp140);
		    X[3 * iostride] = (tmp139 * tmp138) + (tmp137 * tmp140);
	       }
	       {
		    fftw_real tmp144;
		    fftw_real tmp148;
		    fftw_real tmp141;
		    fftw_real tmp145;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp144 = tmp142 + tmp143;
		    tmp148 = tmp146 - tmp147;
		    tmp141 = c_re(W[0]);
		    tmp145 = c_im(W[0]);
		    Y[-8 * iostride] = (tmp141 * tmp144) - (tmp145 * tmp148);
		    X[iostride] = (tmp145 * tmp144) + (tmp141 * tmp148);
	       }
	       {
		    fftw_real tmp150;
		    fftw_real tmp152;
		    fftw_real tmp149;
		    fftw_real tmp151;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp150 = tmp143 - tmp142;
		    tmp152 = tmp146 + tmp147;
		    tmp149 = c_re(W[8]);
		    tmp151 = c_im(W[8]);
		    Y[0] = (tmp149 * tmp150) - (tmp151 * tmp152);
		    X[9 * iostride] = (tmp151 * tmp150) + (tmp149 * tmp152);
	       }
	  }
     }
     if (i == m) {
	  fftw_real tmp1;
	  fftw_real tmp24;
	  fftw_real tmp8;
	  fftw_real tmp10;
	  fftw_real tmp25;
	  fftw_real tmp26;
	  fftw_real tmp14;
	  fftw_real tmp28;
	  fftw_real tmp23;
	  fftw_real tmp17;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp1 = X[2 * iostride];
	  tmp24 = Y[-2 * iostride];
	  {
	       fftw_real tmp2;
	       fftw_real tmp3;
	       fftw_real tmp4;
	       fftw_real tmp5;
	       fftw_real tmp6;
	       fftw_real tmp7;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp2 = X[4 * iostride];
	       tmp3 = X[0];
	       tmp4 = tmp2 + tmp3;
	       tmp5 = X[3 * iostride];
	       tmp6 = X[iostride];
	       tmp7 = tmp5 + tmp6;
	       tmp8 = tmp4 + tmp7;
	       tmp10 = K1_118033988 * (tmp7 - tmp4);
	       tmp25 = tmp2 - tmp3;
	       tmp26 = tmp5 - tmp6;
	  }
	  {
	       fftw_real tmp12;
	       fftw_real tmp13;
	       fftw_real tmp22;
	       fftw_real tmp15;
	       fftw_real tmp16;
	       fftw_real tmp21;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp12 = Y[-4 * iostride];
	       tmp13 = Y[0];
	       tmp22 = tmp12 + tmp13;
	       tmp15 = Y[-iostride];
	       tmp16 = Y[-3 * iostride];
	       tmp21 = tmp16 + tmp15;
	       tmp14 = tmp12 - tmp13;
	       tmp28 = K1_118033988 * (tmp21 + tmp22);
	       tmp23 = tmp21 - tmp22;
	       tmp17 = tmp15 - tmp16;
	  }
	  X[0] = K2_000000000 * (tmp1 + tmp8);
	  {
	       fftw_real tmp18;
	       fftw_real tmp19;
	       fftw_real tmp11;
	       fftw_real tmp20;
	       fftw_real tmp9;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp18 = (K1_175570504 * tmp14) - (K1_902113032 * tmp17);
	       tmp19 = (K1_175570504 * tmp17) + (K1_902113032 * tmp14);
	       tmp9 = (K500000000 * tmp8) - (K2_000000000 * tmp1);
	       tmp11 = tmp9 - tmp10;
	       tmp20 = tmp9 + tmp10;
	       X[2 * iostride] = tmp11 + tmp18;
	       X[8 * iostride] = tmp18 - tmp11;
	       X[4 * iostride] = tmp19 - tmp20;
	       X[6 * iostride] = tmp20 + tmp19;
	  }
	  X[5 * iostride] = K2_000000000 * (tmp23 - tmp24);
	  {
	       fftw_real tmp27;
	       fftw_real tmp31;
	       fftw_real tmp30;
	       fftw_real tmp32;
	       fftw_real tmp29;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp27 = (K1_902113032 * tmp25) + (K1_175570504 * tmp26);
	       tmp31 = (K1_902113032 * tmp26) - (K1_175570504 * tmp25);
	       tmp29 = (K500000000 * tmp23) + (K2_000000000 * tmp24);
	       tmp30 = tmp28 + tmp29;
	       tmp32 = tmp29 - tmp28;
	       X[iostride] = -(tmp27 + tmp30);
	       X[9 * iostride] = tmp27 - tmp30;
	       X[3 * iostride] = tmp31 + tmp32;
	       X[7 * iostride] = tmp32 - tmp31;
	  }
     }
}

static const int twiddle_order[] =
{1, 2, 3, 4, 5, 6, 7, 8, 9};
fftw_codelet_desc fftw_hc2hc_backward_10_desc =
{
     "fftw_hc2hc_backward_10",
     (void (*)()) fftw_hc2hc_backward_10,
     10,
     FFTW_BACKWARD,
     FFTW_HC2HC,
     234,
     9,
     twiddle_order,
};
