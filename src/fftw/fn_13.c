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
/* Generated on Tue May 18 13:54:28 EDT 1999 */

#include <fftw-int.h>
#include <fftw.h>

/* Generated by: ./genfft -magic-alignment-check -magic-twiddle-load-all -magic-variables 4 -magic-loopi -notwiddle 13 */

/*
 * This function contains 176 FP additions, 68 FP multiplications,
 * (or, 138 additions, 30 multiplications, 38 fused multiply/add),
 * 50 stack variables, and 52 memory accesses
 */
static const fftw_real K1_732050807 = FFTW_KONST(+1.732050807568877293527446341505872366942805254);
static const fftw_real K256247671 = FFTW_KONST(+0.256247671582936600958684654061725059144125175);
static const fftw_real K156891391 = FFTW_KONST(+0.156891391051584611046832726756003269660212636);
static const fftw_real K300238635 = FFTW_KONST(+0.300238635966332641462884626667381504676006424);
static const fftw_real K011599105 = FFTW_KONST(+0.011599105605768290721655456654083252189827041);
static const fftw_real K174138601 = FFTW_KONST(+0.174138601152135905005660794929264742616964676);
static const fftw_real K575140729 = FFTW_KONST(+0.575140729474003121368385547455453388461001608);
static const fftw_real K2_000000000 = FFTW_KONST(+2.000000000000000000000000000000000000000000000);
static const fftw_real K083333333 = FFTW_KONST(+0.083333333333333333333333333333333333333333333);
static const fftw_real K075902986 = FFTW_KONST(+0.075902986037193865983102897245103540356428373);
static const fftw_real K251768516 = FFTW_KONST(+0.251768516431883313623436926934233488546674281);
static const fftw_real K258260390 = FFTW_KONST(+0.258260390311744861420450644284508567852516811);
static const fftw_real K132983124 = FFTW_KONST(+0.132983124607418643793760531921092974399165133);
static const fftw_real K265966249 = FFTW_KONST(+0.265966249214837287587521063842185948798330267);
static const fftw_real K387390585 = FFTW_KONST(+0.387390585467617292130675966426762851778775217);
static const fftw_real K503537032 = FFTW_KONST(+0.503537032863766627246873853868466977093348562);
static const fftw_real K113854479 = FFTW_KONST(+0.113854479055790798974654345867655310534642560);
static const fftw_real K300462606 = FFTW_KONST(+0.300462606288665774426601772289207995520941381);
static const fftw_real K866025403 = FFTW_KONST(+0.866025403784438646763723170752936183471402627);
static const fftw_real K500000000 = FFTW_KONST(+0.500000000000000000000000000000000000000000000);

/*
 * Generator Id's : 
 * $Id$
 * $Id$
 * $Id$
 */

void fftw_no_twiddle_13(const fftw_complex *input, fftw_complex *output, int istride, int ostride)
{
     fftw_real tmp1;
     fftw_real tmp88;
     fftw_real tmp29;
     fftw_real tmp36;
     fftw_real tmp43;
     fftw_real tmp121;
     fftw_real tmp128;
     fftw_real tmp30;
     fftw_real tmp24;
     fftw_real tmp131;
     fftw_real tmp124;
     fftw_real tmp129;
     fftw_real tmp41;
     fftw_real tmp44;
     fftw_real tmp134;
     fftw_real tmp83;
     fftw_real tmp89;
     fftw_real tmp70;
     fftw_real tmp85;
     fftw_real tmp137;
     fftw_real tmp141;
     fftw_real tmp146;
     fftw_real tmp77;
     fftw_real tmp86;
     fftw_real tmp144;
     fftw_real tmp147;
     ASSERT_ALIGNED_DOUBLE();
     tmp1 = c_re(input[0]);
     tmp88 = c_im(input[0]);
     {
	  fftw_real tmp15;
	  fftw_real tmp25;
	  fftw_real tmp18;
	  fftw_real tmp26;
	  fftw_real tmp21;
	  fftw_real tmp27;
	  fftw_real tmp22;
	  fftw_real tmp28;
	  fftw_real tmp6;
	  fftw_real tmp37;
	  fftw_real tmp33;
	  fftw_real tmp11;
	  fftw_real tmp38;
	  fftw_real tmp34;
	  fftw_real tmp13;
	  fftw_real tmp14;
	  fftw_real tmp12;
	  fftw_real tmp23;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp13 = c_re(input[8 * istride]);
	  tmp14 = c_re(input[5 * istride]);
	  tmp15 = tmp13 + tmp14;
	  tmp25 = tmp13 - tmp14;
	  {
	       fftw_real tmp16;
	       fftw_real tmp17;
	       fftw_real tmp19;
	       fftw_real tmp20;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp16 = c_re(input[6 * istride]);
	       tmp17 = c_re(input[11 * istride]);
	       tmp18 = tmp16 + tmp17;
	       tmp26 = tmp16 - tmp17;
	       tmp19 = c_re(input[2 * istride]);
	       tmp20 = c_re(input[7 * istride]);
	       tmp21 = tmp19 + tmp20;
	       tmp27 = tmp19 - tmp20;
	  }
	  tmp22 = tmp18 + tmp21;
	  tmp28 = tmp26 + tmp27;
	  {
	       fftw_real tmp2;
	       fftw_real tmp3;
	       fftw_real tmp4;
	       fftw_real tmp5;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp2 = c_re(input[istride]);
	       tmp3 = c_re(input[3 * istride]);
	       tmp4 = c_re(input[9 * istride]);
	       tmp5 = tmp3 + tmp4;
	       tmp6 = tmp2 + tmp5;
	       tmp37 = tmp2 - (K500000000 * tmp5);
	       tmp33 = tmp3 - tmp4;
	  }
	  {
	       fftw_real tmp7;
	       fftw_real tmp8;
	       fftw_real tmp9;
	       fftw_real tmp10;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp7 = c_re(input[12 * istride]);
	       tmp8 = c_re(input[4 * istride]);
	       tmp9 = c_re(input[10 * istride]);
	       tmp10 = tmp8 + tmp9;
	       tmp11 = tmp7 + tmp10;
	       tmp38 = tmp7 - (K500000000 * tmp10);
	       tmp34 = tmp8 - tmp9;
	  }
	  tmp29 = tmp25 - tmp28;
	  {
	       fftw_real tmp32;
	       fftw_real tmp35;
	       fftw_real tmp119;
	       fftw_real tmp120;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp32 = tmp25 + (K500000000 * tmp28);
	       tmp35 = K866025403 * (tmp33 + tmp34);
	       tmp36 = tmp32 - tmp35;
	       tmp43 = tmp35 + tmp32;
	       tmp119 = tmp33 - tmp34;
	       tmp120 = tmp27 - tmp26;
	       tmp121 = tmp119 + tmp120;
	       tmp128 = tmp120 - tmp119;
	  }
	  tmp30 = tmp6 - tmp11;
	  tmp12 = tmp6 + tmp11;
	  tmp23 = tmp15 + tmp22;
	  tmp24 = tmp12 + tmp23;
	  tmp131 = K300462606 * (tmp12 - tmp23);
	  {
	       fftw_real tmp122;
	       fftw_real tmp123;
	       fftw_real tmp39;
	       fftw_real tmp40;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp122 = tmp37 + tmp38;
	       tmp123 = tmp15 - (K500000000 * tmp22);
	       tmp124 = tmp122 + tmp123;
	       tmp129 = tmp122 - tmp123;
	       tmp39 = tmp37 - tmp38;
	       tmp40 = K866025403 * (tmp18 - tmp21);
	       tmp41 = tmp39 + tmp40;
	       tmp44 = tmp39 - tmp40;
	  }
     }
     {
	  fftw_real tmp61;
	  fftw_real tmp135;
	  fftw_real tmp64;
	  fftw_real tmp71;
	  fftw_real tmp67;
	  fftw_real tmp72;
	  fftw_real tmp68;
	  fftw_real tmp136;
	  fftw_real tmp52;
	  fftw_real tmp79;
	  fftw_real tmp75;
	  fftw_real tmp57;
	  fftw_real tmp80;
	  fftw_real tmp74;
	  fftw_real tmp59;
	  fftw_real tmp60;
	  fftw_real tmp139;
	  fftw_real tmp140;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp59 = c_im(input[8 * istride]);
	  tmp60 = c_im(input[5 * istride]);
	  tmp61 = tmp59 + tmp60;
	  tmp135 = tmp59 - tmp60;
	  {
	       fftw_real tmp62;
	       fftw_real tmp63;
	       fftw_real tmp65;
	       fftw_real tmp66;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp62 = c_im(input[6 * istride]);
	       tmp63 = c_im(input[11 * istride]);
	       tmp64 = tmp62 + tmp63;
	       tmp71 = tmp62 - tmp63;
	       tmp65 = c_im(input[2 * istride]);
	       tmp66 = c_im(input[7 * istride]);
	       tmp67 = tmp65 + tmp66;
	       tmp72 = tmp65 - tmp66;
	  }
	  tmp68 = tmp64 + tmp67;
	  tmp136 = tmp71 + tmp72;
	  {
	       fftw_real tmp48;
	       fftw_real tmp49;
	       fftw_real tmp50;
	       fftw_real tmp51;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp48 = c_im(input[istride]);
	       tmp49 = c_im(input[3 * istride]);
	       tmp50 = c_im(input[9 * istride]);
	       tmp51 = tmp49 + tmp50;
	       tmp52 = tmp48 - (K500000000 * tmp51);
	       tmp79 = tmp48 + tmp51;
	       tmp75 = tmp49 - tmp50;
	  }
	  {
	       fftw_real tmp53;
	       fftw_real tmp54;
	       fftw_real tmp55;
	       fftw_real tmp56;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp53 = c_im(input[12 * istride]);
	       tmp54 = c_im(input[4 * istride]);
	       tmp55 = c_im(input[10 * istride]);
	       tmp56 = tmp54 + tmp55;
	       tmp57 = tmp53 - (K500000000 * tmp56);
	       tmp80 = tmp53 + tmp56;
	       tmp74 = tmp54 - tmp55;
	  }
	  tmp134 = tmp79 - tmp80;
	  {
	       fftw_real tmp81;
	       fftw_real tmp82;
	       fftw_real tmp58;
	       fftw_real tmp69;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp81 = tmp79 + tmp80;
	       tmp82 = tmp61 + tmp68;
	       tmp83 = K300462606 * (tmp81 - tmp82);
	       tmp89 = tmp81 + tmp82;
	       tmp58 = tmp52 + tmp57;
	       tmp69 = tmp61 - (K500000000 * tmp68);
	       tmp70 = tmp58 - tmp69;
	       tmp85 = tmp58 + tmp69;
	  }
	  tmp137 = tmp135 - tmp136;
	  tmp139 = K866025403 * (tmp75 + tmp74);
	  tmp140 = tmp135 + (K500000000 * tmp136);
	  tmp141 = tmp139 - tmp140;
	  tmp146 = tmp139 + tmp140;
	  {
	       fftw_real tmp73;
	       fftw_real tmp76;
	       fftw_real tmp142;
	       fftw_real tmp143;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp73 = tmp71 - tmp72;
	       tmp76 = tmp74 - tmp75;
	       tmp77 = tmp73 - tmp76;
	       tmp86 = tmp76 + tmp73;
	       tmp142 = tmp52 - tmp57;
	       tmp143 = K866025403 * (tmp67 - tmp64);
	       tmp144 = tmp142 - tmp143;
	       tmp147 = tmp142 + tmp143;
	  }
     }
     c_re(output[0]) = tmp1 + tmp24;
     {
	  fftw_real tmp163;
	  fftw_real tmp173;
	  fftw_real tmp127;
	  fftw_real tmp169;
	  fftw_real tmp153;
	  fftw_real tmp132;
	  fftw_real tmp138;
	  fftw_real tmp149;
	  fftw_real tmp160;
	  fftw_real tmp172;
	  fftw_real tmp154;
	  fftw_real tmp157;
	  fftw_real tmp158;
	  fftw_real tmp170;
	  fftw_real tmp161;
	  fftw_real tmp162;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp161 = (K113854479 * tmp121) - (K503537032 * tmp124);
	  tmp162 = (K387390585 * tmp128) - (K265966249 * tmp129);
	  tmp163 = tmp161 - tmp162;
	  tmp173 = tmp162 + tmp161;
	  {
	       fftw_real tmp130;
	       fftw_real tmp151;
	       fftw_real tmp125;
	       fftw_real tmp126;
	       fftw_real tmp152;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp130 = (K132983124 * tmp128) + (K258260390 * tmp129);
	       tmp151 = tmp131 - tmp130;
	       tmp125 = (K251768516 * tmp121) + (K075902986 * tmp124);
	       tmp126 = tmp1 - (K083333333 * tmp24);
	       tmp152 = tmp126 - tmp125;
	       tmp127 = (K2_000000000 * tmp125) + tmp126;
	       tmp169 = tmp152 - tmp151;
	       tmp153 = tmp151 + tmp152;
	       tmp132 = (K2_000000000 * tmp130) + tmp131;
	  }
	  {
	       fftw_real tmp145;
	       fftw_real tmp148;
	       fftw_real tmp155;
	       fftw_real tmp156;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp138 = (K575140729 * tmp134) + (K174138601 * tmp137);
	       tmp145 = (K011599105 * tmp141) + (K300238635 * tmp144);
	       tmp148 = (K156891391 * tmp146) - (K256247671 * tmp147);
	       tmp149 = tmp145 + tmp148;
	       tmp160 = K1_732050807 * (tmp148 - tmp145);
	       tmp172 = tmp149 - tmp138;
	       tmp154 = (K174138601 * tmp134) - (K575140729 * tmp137);
	       tmp155 = (K300238635 * tmp141) - (K011599105 * tmp144);
	       tmp156 = (K256247671 * tmp146) + (K156891391 * tmp147);
	       tmp157 = tmp155 + tmp156;
	       tmp158 = tmp154 - tmp157;
	       tmp170 = K1_732050807 * (tmp156 - tmp155);
	  }
	  {
	       fftw_real tmp133;
	       fftw_real tmp150;
	       fftw_real tmp165;
	       fftw_real tmp166;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp133 = tmp127 - tmp132;
	       tmp150 = tmp138 + (K2_000000000 * tmp149);
	       c_re(output[8 * ostride]) = tmp133 - tmp150;
	       c_re(output[5 * ostride]) = tmp133 + tmp150;
	       {
		    fftw_real tmp167;
		    fftw_real tmp168;
		    fftw_real tmp159;
		    fftw_real tmp164;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp167 = tmp132 + tmp127;
		    tmp168 = tmp154 + (K2_000000000 * tmp157);
		    c_re(output[12 * ostride]) = tmp167 - tmp168;
		    c_re(output[ostride]) = tmp167 + tmp168;
		    tmp159 = tmp153 - tmp158;
		    tmp164 = tmp160 - tmp163;
		    c_re(output[4 * ostride]) = tmp159 - tmp164;
		    c_re(output[10 * ostride]) = tmp164 + tmp159;
	       }
	       tmp165 = tmp153 + tmp158;
	       tmp166 = tmp163 + tmp160;
	       c_re(output[3 * ostride]) = tmp165 - tmp166;
	       c_re(output[9 * ostride]) = tmp166 + tmp165;
	       {
		    fftw_real tmp175;
		    fftw_real tmp176;
		    fftw_real tmp171;
		    fftw_real tmp174;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp175 = tmp169 + tmp170;
		    tmp176 = tmp173 + tmp172;
		    c_re(output[2 * ostride]) = tmp175 - tmp176;
		    c_re(output[7 * ostride]) = tmp176 + tmp175;
		    tmp171 = tmp169 - tmp170;
		    tmp174 = tmp172 - tmp173;
		    c_re(output[6 * ostride]) = tmp171 - tmp174;
		    c_re(output[11 * ostride]) = tmp174 + tmp171;
	       }
	  }
     }
     c_im(output[0]) = tmp88 + tmp89;
     {
	  fftw_real tmp102;
	  fftw_real tmp115;
	  fftw_real tmp84;
	  fftw_real tmp112;
	  fftw_real tmp106;
	  fftw_real tmp91;
	  fftw_real tmp31;
	  fftw_real tmp46;
	  fftw_real tmp107;
	  fftw_real tmp111;
	  fftw_real tmp94;
	  fftw_real tmp97;
	  fftw_real tmp99;
	  fftw_real tmp114;
	  fftw_real tmp100;
	  fftw_real tmp101;
	  ASSERT_ALIGNED_DOUBLE();
	  tmp100 = (K387390585 * tmp77) + (K265966249 * tmp70);
	  tmp101 = (K113854479 * tmp86) + (K503537032 * tmp85);
	  tmp102 = tmp100 + tmp101;
	  tmp115 = tmp100 - tmp101;
	  {
	       fftw_real tmp78;
	       fftw_real tmp105;
	       fftw_real tmp87;
	       fftw_real tmp90;
	       fftw_real tmp104;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp78 = (K258260390 * tmp70) - (K132983124 * tmp77);
	       tmp105 = tmp83 - tmp78;
	       tmp87 = (K075902986 * tmp85) - (K251768516 * tmp86);
	       tmp90 = tmp88 - (K083333333 * tmp89);
	       tmp104 = tmp90 - tmp87;
	       tmp84 = (K2_000000000 * tmp78) + tmp83;
	       tmp112 = tmp105 + tmp104;
	       tmp106 = tmp104 - tmp105;
	       tmp91 = (K2_000000000 * tmp87) + tmp90;
	  }
	  {
	       fftw_real tmp42;
	       fftw_real tmp45;
	       fftw_real tmp95;
	       fftw_real tmp96;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp31 = (K575140729 * tmp29) - (K174138601 * tmp30);
	       tmp42 = (K300238635 * tmp36) + (K011599105 * tmp41);
	       tmp45 = (K256247671 * tmp43) + (K156891391 * tmp44);
	       tmp46 = tmp42 - tmp45;
	       tmp107 = K1_732050807 * (tmp45 + tmp42);
	       tmp111 = tmp31 - tmp46;
	       tmp94 = (K575140729 * tmp30) + (K174138601 * tmp29);
	       tmp95 = (K156891391 * tmp43) - (K256247671 * tmp44);
	       tmp96 = (K300238635 * tmp41) - (K011599105 * tmp36);
	       tmp97 = tmp95 + tmp96;
	       tmp99 = tmp97 - tmp94;
	       tmp114 = K1_732050807 * (tmp96 - tmp95);
	  }
	  {
	       fftw_real tmp47;
	       fftw_real tmp92;
	       fftw_real tmp109;
	       fftw_real tmp110;
	       ASSERT_ALIGNED_DOUBLE();
	       tmp47 = tmp31 + (K2_000000000 * tmp46);
	       tmp92 = tmp84 + tmp91;
	       c_im(output[ostride]) = tmp47 + tmp92;
	       c_im(output[12 * ostride]) = tmp92 - tmp47;
	       {
		    fftw_real tmp93;
		    fftw_real tmp98;
		    fftw_real tmp103;
		    fftw_real tmp108;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp93 = tmp91 - tmp84;
		    tmp98 = tmp94 + (K2_000000000 * tmp97);
		    c_im(output[5 * ostride]) = tmp93 - tmp98;
		    c_im(output[8 * ostride]) = tmp98 + tmp93;
		    tmp103 = tmp99 + tmp102;
		    tmp108 = tmp106 - tmp107;
		    c_im(output[2 * ostride]) = tmp103 + tmp108;
		    c_im(output[7 * ostride]) = tmp108 - tmp103;
	       }
	       tmp109 = tmp107 + tmp106;
	       tmp110 = tmp102 - tmp99;
	       c_im(output[6 * ostride]) = tmp109 - tmp110;
	       c_im(output[11 * ostride]) = tmp110 + tmp109;
	       {
		    fftw_real tmp117;
		    fftw_real tmp118;
		    fftw_real tmp113;
		    fftw_real tmp116;
		    ASSERT_ALIGNED_DOUBLE();
		    tmp117 = tmp112 - tmp111;
		    tmp118 = tmp114 - tmp115;
		    c_im(output[4 * ostride]) = tmp117 - tmp118;
		    c_im(output[10 * ostride]) = tmp118 + tmp117;
		    tmp113 = tmp111 + tmp112;
		    tmp116 = tmp114 + tmp115;
		    c_im(output[3 * ostride]) = tmp113 - tmp116;
		    c_im(output[9 * ostride]) = tmp116 + tmp113;
	       }
	  }
     }
}

fftw_codelet_desc fftw_no_twiddle_13_desc =
{
     "fftw_no_twiddle_13",
     (void (*)()) fftw_no_twiddle_13,
     13,
     FFTW_FORWARD,
     FFTW_NOTW,
     287,
     0,
     (const int *) 0,
};
