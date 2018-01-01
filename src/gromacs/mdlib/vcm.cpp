/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
/* This file is completely threadsafe - keep it that way! */
#include "gmxpre.h"

#include "gromacs/legacyheaders/vcm.h"

#include "gromacs/legacyheaders/macros.h"
#include "gromacs/legacyheaders/names.h"
#include "gromacs/legacyheaders/network.h"
#include "gromacs/legacyheaders/txtdump.h"
#include "gromacs/math/vec.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/utility/smalloc.h"
/* #include "mvdata.h" */    /* Required for applying RTC in parallel */
#include "gromacs/legacyheaders/gmx_ga2la.h" /* Required for applying RTC in parallel */
#include "gromacs/fileio/confio.h"
#include "gromacs/topology/mtop_util.h"
 
/* We use the same defines as in gctio.c here */
/* Shouldn't these be put in mvdata.h? They're now in three different places. */

#define  block_bc(cr,   d) gmx_bcast(     sizeof(d),     &(d),(cr))
#define nblock_bc(cr,nr,d) gmx_bcast((nr)*sizeof((d)[0]), (d),(cr))
#define   snew_bc(cr,d,nr) { if (!MASTER(cr)) snew((d),(nr)); }

static void outer_inc(rvec x,rvec y,rvec z)
{
  z[0] += x[1]*y[2]-x[2]*y[1];
  z[1] += x[2]*y[0]-x[0]*y[2];
  z[2] += x[0]*y[1]-x[1]*y[0];
}

t_vcm *init_vcm(FILE *fp, gmx_groups_t *groups, t_inputrec *ir)
{
    t_vcm *vcm;
    int    g;

    snew(vcm, 1);

    vcm->mode = (ir->nstcomm > 0) ? ir->comm_mode : ecmNO;
    vcm->ndim = ndof_com(ir);

    if (vcm->mode == ecmANGULAR && vcm->ndim < 3)
    {
        gmx_fatal(FARGS, "Can not have angular comm removal with pbc=%s",
                  epbc_names[ir->ePBC]);
    }

    if (vcm->mode != ecmNO)
    {
        vcm->nr = groups->grps[egcVCM].nr;
        /* Allocate one extra for a possible rest group */
        if (vcm->mode == ecmANGULAR)
        {
            snew(vcm->group_j, vcm->nr+1);
            snew(vcm->group_x, vcm->nr+1);
            snew(vcm->group_i, vcm->nr+1);
            snew(vcm->group_w, vcm->nr+1);
        }
        snew(vcm->group_p, vcm->nr+1);
        snew(vcm->group_v, vcm->nr+1);
        snew(vcm->group_mass, vcm->nr+1);
        snew(vcm->group_name, vcm->nr);
        snew(vcm->group_ndf, vcm->nr);
        for (g = 0; (g < vcm->nr); g++)
        {
            vcm->group_ndf[g] = ir->opts.nrdf[g];
        }

        /* Copy pointer to group names and print it. */
        if (fp)
        {
            fprintf(fp, "Center of mass motion removal mode is %s\n",
                    ECOM(vcm->mode));
            fprintf(fp, "We have the following groups for center of"
                    " mass motion removal:\n");
        }
        for (g = 0; (g < vcm->nr); g++)
        {
            vcm->group_name[g] = *groups->grpname[groups->grps[egcVCM].nm_ind[g]];
            if (fp)
            {
                fprintf(fp, "%3d:  %s\n", g, vcm->group_name[g]);
            }
        }
    }

    vcm->rtc = NULL;

    return vcm;
}

static void update_tensor(rvec x, real m0, tensor I)
{
    real xy, xz, yz;

    /* Compute inertia tensor contribution due to this atom */
    xy         = x[XX]*x[YY]*m0;
    xz         = x[XX]*x[ZZ]*m0;
    yz         = x[YY]*x[ZZ]*m0;
    I[XX][XX] += x[XX]*x[XX]*m0;
    I[YY][YY] += x[YY]*x[YY]*m0;
    I[ZZ][ZZ] += x[ZZ]*x[ZZ]*m0;
    I[XX][YY] += xy;
    I[YY][XX] += xy;
    I[XX][ZZ] += xz;
    I[ZZ][XX] += xz;
    I[YY][ZZ] += yz;
    I[ZZ][YY] += yz;
}

/* Center of mass code for groups */
void calc_vcm_grp(FILE *fp, int *la2ga, int start, int homenr, t_mdatoms *md,
		  rvec x[], rvec v[], t_vcm *vcm)
{
    int    i, j, g, gi, m;
    real   m0, xx, xy, xz, yy, yz, zz;
    rvec   j0, d;
    t_rtc  *rtc=NULL;
    
    rtc = vcm->rtc;
    
    if (vcm->mode != ecmNO) 
    {
        /* Also clear a possible rest group */
        for (g = 0; (g < vcm->nr+1); g++)
        {
            /* Reset linear momentum */
            vcm->group_mass[g] = 0;
            clear_rvec(vcm->group_p[g]);

            if (vcm->mode == ecmANGULAR)
            {
                /* Reset anular momentum */
                clear_rvec(vcm->group_j[g]);
                clear_rvec(vcm->group_x[g]);
                clear_rvec(vcm->group_w[g]);
                clear_mat(vcm->group_i[g]);
            }
            if (vcm->mode == ecmRTC && g < rtc->nr)
            {
                clear_rvec(rtc->outerx[g]);
                clear_rvec(rtc->sumx[g]); 
                clear_rvec(rtc->outerv[g]);
                clear_rvec(rtc->sumv[g]); 
                if (0 && md->nMassPerturbed)
                {
                    clear_rvec(rtc->refcom[g]);
                }
            }
        }

        g = 0;
        for (i = start; (i < start+homenr); i++)
        {
            m0 = md->massT[i];
            if (md->cVCM)
            {
                g = md->cVCM[i];
            }

            /* Calculate linear momentum */
            vcm->group_mass[g]  += m0;
            for (m = 0; (m < DIM); m++)
            {
                vcm->group_p[g][m] += m0*v[i][m];
            }

            if (vcm->mode == ecmANGULAR)
            {
                /* Calculate angular momentum */
                cprod(x[i], v[i], j0);

                for (m = 0; (m < DIM); m++)
                {
                    vcm->group_j[g][m] += m0*j0[m];
                    vcm->group_x[g][m] += m0*x[i][m];
                }
                /* Update inertia tensor */
                update_tensor(x[i], m0, vcm->group_i[g]);
            }
            else if (vcm->mode == ecmRTC)
            {
                gi = la2ga ? la2ga[i] : i;
                
                if (g >= rtc->nr)
                    continue;

                /* 
                    RTC stuff from velocities
                    Correction is applied to current positions based on past velocities,
                    and to current velocities (t or t+dt/2) based on these velocities.
                    The vectors sumv, outerv, sumx and outerx need to be communicated/collected.
                 */
                svmul(m0,v[i],d);
                outer_inc(d,rtc->xref[gi],rtc->outerv[g]);
                rvec_inc(rtc->sumv[g],d);

                if (0 && md->nMassPerturbed)
                {
                    svmul(m0,rtc->xref[gi],d);
                    rvec_inc(rtc->refcom[g],d);
                }
            }
        }

        if (vcm->mode == ecmRTC && rtc->nst > 1)
        {
	    for (g=0; g<rtc->nr; g++)
	    {
	        rvec_inc(rtc->sumc[g],   rtc->sumv[g]);
	        rvec_inc(rtc->outerc[g], rtc->outerv[g]);

                /* Positional correction from velocity depends on time step */
                svmul(rtc->dt, rtc->sumc[g],   rtc->sumx[g]);
                svmul(rtc->dt, rtc->outerc[g], rtc->outerx[g]);
	    }
	}
    }
}

void do_stopcm_grp(FILE *fp, int *la2ga, int start, int homenr, unsigned short *group_id,
                   rvec x[], rvec v[], t_vcm *vcm)
{
    int   i, j, gi, g, m;
    real  tm, tm_1, c;
    rvec  dv, dvc, dvt, dx, d;
    t_rtc *rtc=NULL;
    rvec  *axisx=NULL, *outerx=NULL, *axisv=NULL, *outerv=NULL, *shiftx=NULL, *shiftv=NULL;

    if (vcm->mode == ecmRTC)
    {
        rtc = vcm->rtc;

	/*
             For the derivation of this check:

             ...
             ...
        
             Correction per atom

             x_i'      = x_i + c_i
             v_i'      = v_i + c_i / dt
         
             c_i / dt  = u (x) (r_i - r_com) - p / sum{m}
                       = u (x) r_i - ( u (x) r_com + p / sum{m} )
                         ~~~~~~~~~   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                         per atom            per group
                       = u (x) r_i - s

             u         = minv(I_r) . (q + r_com (x) p)
             s         = u (x) r_com + p / sum{m}

             Accumulated over k steps:
             p_x       = sum_k{ sum_j{ m_j * v_jk } }          -- rtc->sumc[g]    
             q_x       = sum_k{ sum_j{ m_j * v_jk (x) r_j } }  -- rtc->outerc[g]

             Last step only:
             p_v       = sum_j{ m_j * v_j }                    -- rtc->sumv[g]
             q_v       = sum_j{ m_j * v_j (x) r_j }            -- rtc->outerv[g]
 	 */

        snew(axisx,  rtc->nr);
        snew(outerx, rtc->nr);
        snew(shiftx, rtc->nr);

        snew(axisv,  rtc->nr);
        snew(outerv, rtc->nr);
        snew(shiftv, rtc->nr);

        for (g=0; g < rtc->nr; g++)
        {
            /* Determine the sort-of-axes-of-rotation */

            /* Rotation axis for velocities */
	    outer_inc(rtc->refcom[g], rtc->sumv[g], rtc->outerv[g]);
            mvmul(rtc->invinert[g], rtc->outerv[g], axisv[g]);   /* u_v */

            /* Shift for velocities */
            svmul(1.0/vcm->group_mass[g], rtc->sumv[g], shiftv[g]);
            outer_inc(axisv[g], rtc->refcom[g], shiftv[g]);
            
            /* Rotation axis for positions */
	    outer_inc(rtc->refcom[g], rtc->sumx[g], rtc->outerx[g]);
            mvmul(rtc->invinert[g], rtc->outerx[g], axisx[g]);   /* u_x */

            /* Shift for positions */
            svmul(1.0/vcm->group_mass[g], rtc->sumx[g], shiftx[g]);
            outer_inc(axisx[g], rtc->refcom[g], shiftx[g]);

        }

	if (rtc->log)
	{
	    rtc->time += rtc->nst * rtc->dt;
  	    fprintf(rtc->log, "%10.5f ", rtc->time);
	    for (g=0; g<rtc->nr; g++)
	    {
  	        fprintf(rtc->log, "%g %g %g %g %g %g %g %g %g ",
			axisx[g][0], axisx[g][1], axisx[g][2],
			outerx[g][0], outerx[g][1], outerx[g][2],
			shiftx[g][0], shiftx[g][1], shiftx[g][2]
			);
		fprintf(rtc->log, "%g %g %g %g %g %g %g %g %g ",
			axisv[g][0], axisv[g][1], axisv[g][2],
			outerv[g][0], outerv[g][1], outerv[g][2],
			shiftv[g][0], shiftv[g][1], shiftv[g][2]
			);
	    }
	    fprintf(rtc->log, "\n");
	}

        /* Determine per particle correction */
        g = 0;
        for (i = start; i < homenr; i++)
        {
            if (group_id && (g = group_id[i]) >= rtc->nr)
            {
                continue;
            }
            gi = la2ga ? la2ga[i] : i;

            /* Subtract COM of reference group from reference coordinate */
            /* rvec_sub(rtc->xref[gi], rtc->refcom[g], d); */

            /* Velocity correction follows from cross product *
             * of group axis with reference coordinate        */
	    outer_inc(axisv[g], rtc->xref[gi], v[i]);
	    rvec_dec(v[i], shiftv[g]);
	    outer_inc(axisx[g], rtc->xref[gi], x[i]);
	    rvec_dec(x[i], shiftx[g]);
        }

	for (g=0; g<rtc->nr; g++)
	{
	    clear_rvec(rtc->sumc[g]);
	    clear_rvec(rtc->sumx[g]);
	    clear_rvec(rtc->sumv[g]);
	    clear_rvec(rtc->outerc[g]);
	    clear_rvec(rtc->outerx[g]);
	    clear_rvec(rtc->outerv[g]);	  
	}
        sfree(axisx);
        sfree(axisv);
	sfree(shiftx);
	sfree(shiftv);
    }
    else if (vcm->mode != ecmNO) 
    {
        /* Subtract linear momentum */
        g = 0;
        switch (vcm->ndim)
        {
            case 1:
                for (i = start; (i < start+homenr); i++)
                {
                    if (group_id)
                    {
                        g = group_id[i];
                    }
                    v[i][XX] -= vcm->group_v[g][XX];
                }
                break;
            case 2:
                for (i = start; (i < start+homenr); i++)
                {
                    if (group_id)
                    {
                        g = group_id[i];
                    }
                    v[i][XX] -= vcm->group_v[g][XX];
                    v[i][YY] -= vcm->group_v[g][YY];
                }
                break;
            case 3:
                for (i = start; (i < start+homenr); i++)
                {
                    if (group_id)
                    {
                        g = group_id[i];
                    }
                    rvec_dec(v[i], vcm->group_v[g]);
                }
                break;
        }
        if (vcm->mode == ecmANGULAR)
        {
            /* Subtract angular momentum */
            for (i = start; (i < start+homenr); i++)
            {
                if (group_id)
                {
                    g = group_id[i];
                }
                /* Compute the correction to the velocity for each atom */
                rvec_sub(x[i], vcm->group_x[g], dx);
                cprod(vcm->group_w[g], dx, dv);
                rvec_dec(v[i], dv);
            }
        }
    }
}

static void get_minv(tensor A, tensor B)
{
    int    m, n;
    double fac, rfac;
    tensor tmp;

    tmp[XX][XX] =  A[YY][YY] + A[ZZ][ZZ];
    tmp[YY][XX] = -A[XX][YY];
    tmp[ZZ][XX] = -A[XX][ZZ];
    tmp[XX][YY] = -A[XX][YY];
    tmp[YY][YY] =  A[XX][XX] + A[ZZ][ZZ];
    tmp[ZZ][YY] = -A[YY][ZZ];
    tmp[XX][ZZ] = -A[XX][ZZ];
    tmp[YY][ZZ] = -A[YY][ZZ];
    tmp[ZZ][ZZ] =  A[XX][XX] + A[YY][YY];

    /* This is a hack to prevent very large determinants */
    rfac  = (tmp[XX][XX]+tmp[YY][YY]+tmp[ZZ][ZZ])/3;
    if (rfac == 0.0)
    {
        gmx_fatal(FARGS, "Can not stop center of mass: maybe 2dimensional system");
    }
    fac = 1.0/rfac;
    for (m = 0; (m < DIM); m++)
    {
        for (n = 0; (n < DIM); n++)
        {
            tmp[m][n] *= fac;
        }
    }
    m_inv(tmp, B);
    for (m = 0; (m < DIM); m++)
    {
        for (n = 0; (n < DIM); n++)
        {
            B[m][n] *= fac;
        }
    }
}

void check_cm_grp(FILE *fp, t_vcm *vcm, t_inputrec *ir, real Temp_Max)
{
    int    m, g;
    real   ekcm, ekrot, tm, tm_1, Temp_cm;
    rvec   jcm;
    tensor Icm;

    /* First analyse the total results */
    if (vcm->mode != ecmNO)
    {
        for (g = 0; (g < vcm->nr); g++)
        {
            tm = vcm->group_mass[g];
            if (tm != 0)
            {
                tm_1 = 1.0/tm;
                svmul(tm_1, vcm->group_p[g], vcm->group_v[g]);
            }
            /* Else it's zero anyway! */
        }
        if (vcm->mode == ecmANGULAR)
        {
            for (g = 0; (g < vcm->nr); g++)
            {
                tm = vcm->group_mass[g];
                if (tm != 0)
                {
                    tm_1 = 1.0/tm;

                    /* Compute center of mass for this group */
                    for (m = 0; (m < DIM); m++)
                    {
                        vcm->group_x[g][m] *= tm_1;
                    }

                    /* Subtract the center of mass contribution to the
                     * angular momentum
                     */
                    cprod(vcm->group_x[g], vcm->group_v[g], jcm);
                    for (m = 0; (m < DIM); m++)
                    {
                        vcm->group_j[g][m] -= tm*jcm[m];
                    }

                    /* Subtract the center of mass contribution from the inertia
                     * tensor (this is not as trivial as it seems, but due to
                     * some cancellation we can still do it, even in parallel).
                     */
                    clear_mat(Icm);
                    update_tensor(vcm->group_x[g], tm, Icm);
                    m_sub(vcm->group_i[g], Icm, vcm->group_i[g]);

                    /* Compute angular velocity, using matrix operation
                     * Since J = I w
                     * we have
                     * w = I^-1 J
                     */
                    get_minv(vcm->group_i[g], Icm);
                    mvmul(Icm, vcm->group_j[g], vcm->group_w[g]);
                }
                /* Else it's zero anyway! */
            }
        }
    }
    for (g = 0; (g < vcm->nr); g++)
    {
        ekcm    = 0;
        if (vcm->group_mass[g] != 0 && vcm->group_ndf[g] > 0)
        {
            for (m = 0; m < vcm->ndim; m++)
            {
                ekcm += sqr(vcm->group_v[g][m]);
            }
            ekcm   *= 0.5*vcm->group_mass[g];
            Temp_cm = 2*ekcm/vcm->group_ndf[g];

            if ((Temp_cm > Temp_Max) && fp)
            {
                fprintf(fp, "Large VCM(group %s): %12.5f, %12.5f, %12.5f, Temp-cm: %12.5e\n",
                        vcm->group_name[g], vcm->group_v[g][XX],
                        vcm->group_v[g][YY], vcm->group_v[g][ZZ], Temp_cm);
            }

            if (vcm->mode == ecmANGULAR)
            {
                ekrot = 0.5*iprod(vcm->group_j[g], vcm->group_w[g]);
                if ((ekrot > 1) && fp && !EI_RANDOM(ir->eI))
                {
                    /* if we have an integrator that may not conserve momenta, skip */
                    tm    = vcm->group_mass[g];
                    fprintf(fp, "Group %s with mass %12.5e, Ekrot %12.5e Det(I) = %12.5e\n",
                            vcm->group_name[g], tm, ekrot, det(vcm->group_i[g]));
                    fprintf(fp, "  COM: %12.5f  %12.5f  %12.5f\n",
                            vcm->group_x[g][XX], vcm->group_x[g][YY], vcm->group_x[g][ZZ]);
                    fprintf(fp, "  P:   %12.5f  %12.5f  %12.5f\n",
                            vcm->group_p[g][XX], vcm->group_p[g][YY], vcm->group_p[g][ZZ]);
                    fprintf(fp, "  V:   %12.5f  %12.5f  %12.5f\n",
                            vcm->group_v[g][XX], vcm->group_v[g][YY], vcm->group_v[g][ZZ]);
                    fprintf(fp, "  J:   %12.5f  %12.5f  %12.5f\n",
                            vcm->group_j[g][XX], vcm->group_j[g][YY], vcm->group_j[g][ZZ]);
                    fprintf(fp, "  w:   %12.5f  %12.5f  %12.5f\n",
                            vcm->group_w[g][XX], vcm->group_w[g][YY], vcm->group_w[g][ZZ]);
                    pr_rvecs(fp, 0, "Inertia tensor", vcm->group_i[g], DIM);
                }
            }
        }
    }
}

/*
 * Roto-translational constraints
 *
 * 
 */

t_rtc *init_rtc(gmx_mtop_t  *mtop,   /* global topology                     */
                t_mdatoms   *atoms,  /* atom stuff; need masses             */
                t_commrec   *cr,     /* communication record                */
                t_inputrec  *ir,     /* input record                        */
                const char  *fnRTC,  /* file containing reference structure */
                const char  *fnLOG,  /* file for logging/debugging          */
                rvec        *x,      /* positions of the whole MD system;   */
                const char  *fnTPX)  /* required if we read in a checkpoint */
{
    int     natoms,epbc,i,j,k,g;
    char    title[STRLEN];
    t_atoms refatoms;
    t_atom  *atom;
    t_state start_state;
    gmx_mtop_atomlookup_t alook;
    matrix  box,mi,T,S;
    real    m0,*tm;
    rvec    di,ri;

    t_rtc   *rtc;

    gmx_groups_t *groups=&mtop->groups;

    clear_mat(S);
    clear_mat(T);
    
    /* Set stuff for RTC groups */
    snew(rtc, 1);
    rtc->nr = groups->grps[egcVCM].nr;
    /*
    fprintf(stderr, "%d RTC groups -> ", rtc->nr);
    if (rtc->nr > 1)
    {
        rtc->nr--;
    }
    fprintf(stderr, "%d RTC groups\n", rtc->nr);
    */
    
    snew(rtc->refcom,   rtc->nr+1);
    snew(rtc->invinert, rtc->nr+1);    
    snew(rtc->outerx,   rtc->nr+1);
    snew(rtc->outerv,   rtc->nr+1);
    snew(rtc->outerc,   rtc->nr+1);
    snew(rtc->sumx,     rtc->nr+1);
    snew(rtc->sumv,     rtc->nr+1); 
    snew(rtc->sumc,     rtc->nr+1); 
    snew(tm,            rtc->nr+1);
    for (g=0; g < rtc->nr+1; g++)
    {
        clear_rvec(rtc->refcom[g]);
        clear_rvec(rtc->outerx[g]);
        clear_rvec(rtc->outerv[g]);
        clear_rvec(rtc->outerc[g]);
        clear_rvec(rtc->sumx[g]);
        clear_rvec(rtc->sumv[g]);
        clear_rvec(rtc->sumc[g]);
        clear_mat(rtc->invinert[g]);
    }
    rtc->xp   = NULL;
    rtc->nst  = ir->nstcomm;
    rtc->dt   = ir->delta_t;
    rtc->time = ir->init_t; 

    rtc->log = NULL;
    
    /* Read in the reference file if one is given */
    if (MASTER(cr))
    {
        if (fnLOG)
	{
	    rtc->log = gmx_ffopen(fnLOG, "w");
	    fprintf(stderr, "Logging RTC correction per step to %s\n", fnLOG);
	}
	
        read_tpx_state(fnTPX, NULL, &start_state, NULL, NULL);

        fprintf(stderr,"Reading RTC reference coordinates from %s\n", fnRTC ? fnRTC : fnTPX);

        if (fnRTC)
        {
            get_stx_coordnum(fnRTC, &(rtc->nref));
            init_t_atoms(&refatoms,rtc->nref,FALSE);
            snew(rtc->xref,rtc->nref);
            read_stx_conf(fnRTC, title, &refatoms, rtc->xref, NULL, &epbc, box );        
        }
        else
        {
            rtc->nref = mtop->natoms;
            snew(rtc->xref, mtop->natoms);
            for (i=0; i < mtop->natoms; i++)
            {
                copy_rvec(start_state.x[i], rtc->xref[i]);
            }            
        }
    }

    /* Now have to communicate the whole lot */
    if (PAR(cr))
    {
        block_bc(  cr, rtc->nref           );
        snew_bc(   cr, rtc->xref, rtc->nref);
        nblock_bc( cr, rtc->nref, rtc->xref);
    }

    alook = gmx_mtop_atomlookup_init(mtop);

    /* Then calculate the COM and matrix of inertia per group */
    g = 0;
    for (i = 0; i < rtc->nref; i++)
    {
        if (groups->grpnr[egcVCM])
        {
            g = groups->grpnr[egcVCM][i];
        }
        gmx_mtop_atomnr_to_atom(alook, i, &atom);
        m0 = atom->m;
        tm[g] += m0;
        for (j=0; j<DIM; j++)
        {
            rtc->refcom[g][j] += m0*rtc->xref[i][j];

            /* Whoever decides to set DIM to something else than 3 
             * has to make this a double for loop :p
             */
            k = (j+1)%DIM;
            rtc->invinert[g][j][j] += m0*rtc->xref[i][j]*rtc->xref[i][j];
            rtc->invinert[g][j][k] += m0*rtc->xref[i][j]*rtc->xref[i][k];
        }
    }

    gmx_mtop_atomlookup_destroy(alook);

    for (g=0; g < rtc->nr; g++)
    {
        svmul( 1.0/tm[g], rtc->refcom[g], rtc->refcom[g] );
        di[0] = rtc->refcom[g][0]*rtc->refcom[g][0];
        di[1] = rtc->refcom[g][1]*rtc->refcom[g][1];
        di[2] = rtc->refcom[g][2]*rtc->refcom[g][2];
        for (i=0; i<DIM; i++)
        {
            j = (i+1)%DIM;
            k = (i+2)%DIM;
            mi[i][i] = rtc->invinert[g][j][j] + rtc->invinert[g][k][k] - tm[g]*(di[j]+di[k]);
            mi[i][j] = mi[j][i] = tm[g]*rtc->refcom[g][i]*rtc->refcom[g][j] - rtc->invinert[g][i][j];
        }
        m_inv(mi, rtc->invinert[g]);
    }

    return rtc;
}

void purge_rtc(FILE *fp, int *la2ga, t_mdatoms *md, rvec v[], t_rtc *rtc, gmx_bool bStopCM)
{
    int    i, j, g, gi;
    real   m0;
    rvec   d, oc, sc;

    if (bStopCM)
    {
        /* We just applied corrections: clear data */
        for (g=0; g<rtc->nr; g++)
	{
	    clear_rvec(rtc->sumc[g]);
	    clear_rvec(rtc->outerc[g]);
	}

        /* The velocities have been corrected, so no rotation or com motion this step */
        return;
    }

    /* Accumulate data for the next round */
    g  = 0;
    for (i = 0; i < md->homenr; i++) 
    {
        m0 = md->massT[i];

        if (md->cVCM)
            g = md->cVCM[i];
        
        gi = la2ga ? la2ga[i] : i;

        if (g >= rtc->nr || gi >= rtc->nref)
        {
            continue;
        }

        /* Positions */
        svmul(m0, v[i], d);
        outer_inc(d, rtc->xref[gi], rtc->outerc[g]);
        rvec_inc(rtc->sumc[g], d);
    }
}

