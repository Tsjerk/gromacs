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

#ifndef _vcm_h
#define _vcm_h

#include <stdio.h>

/*
#include "gromacs/fileio/filenm.h"
#include "gromacs/legacyheaders/mdebin.h"
#include "gromacs/legacyheaders/sim_util.h"
#include "gromacs/legacyheaders/tgroup.h"
#include "gromacs/legacyheaders/update.h"
#include "gromacs/legacyheaders/vsite.h"
#include "gromacs/legacyheaders/types/membedt.h"
#include "gromacs/timing/wallcycle.h"
*/

#include "gromacs/legacyheaders/typedefs.h"
#include "gromacs/legacyheaders/network.h"

#include "gromacs/legacyheaders/types/inputrec.h"
#include "gromacs/legacyheaders/types/mdatom.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Roto-translational constraints */
typedef struct {
  /* Set at initialization */
  int    nr;                   /* Number of groups                       */
  int    nref;                 /* Number of atoms in reference structure */
  rvec   *xref;                /* Reference structure                    */
  rvec   *xp;                  /* Previous coordinates                   */
  matrix *invinert;            /* Inverse matrix of inertia per group    */
  gmx_bool   bFEP;             /* Set when doing FEP (masses may change) */
  rvec   *refcom;              /* COM per group
				  Note that masses may change during a
				  simulation prohibiting precalculation  */
  FILE   *log;                 /* File for logging/debugging             */

  int    nst;                  /* Frequency (steps) for update (nstcomm) */
  real   dt;                   /* Integration time step                  */
  double time;                 /* Time for logging                       */

  rvec   *outerc;              /* Cumulative some of cross products      */
  rvec   *sumc;                /* Cumulative some of deviations          */

  /* Only the following require global communication 
   * and only at every nstcomm steps                                     */
  rvec   *outerx;              /* Sum of cross products of deviations
				  and reference positions                */
  rvec   *outerv;              /* Sum of cross products of velocities
				  and reference positions                */
  rvec   *sumx;                /* Sum of m*x per group                   */
  rvec   *sumv;                /* Sum of m*v per group                   */
} t_rtc;

struct gmx_groups_t;

typedef struct {
    int        nr;             /* Number of groups                    */
    int        mode;           /* One of the enums above              */
    gmx_bool   ndim;           /* The number of dimensions for corr.  */
    real      *group_ndf;      /* Number of degrees of freedom        */
    rvec      *group_p;        /* Linear momentum per group           */
    rvec      *group_v;        /* Linear velocity per group           */
    rvec      *group_x;        /* Center of mass per group            */
    rvec      *group_j;        /* Angular momentum per group          */
    rvec      *group_w;        /* Angular velocity (omega)            */
    tensor    *group_i;        /* Moment of inertia per group         */
    real      *group_mass;     /* Mass per group                      */
    char     **group_name;     /* These two are copies to pointers in */
    t_rtc     *rtc;            /* Rotational constraints stuff        */
} t_vcm;

t_vcm *init_vcm(FILE *fp, struct gmx_groups_t *groups, t_inputrec *ir);

t_rtc *init_rtc(gmx_mtop_t *mtop, t_mdatoms *atoms, t_commrec *cr, t_inputrec *ir,
		const char *fnRTC, const char *fnLOG, rvec *x, const char *fnTPR);

extern void purge_rtc(FILE *fp,int *la2ga,t_mdatoms *md,rvec x[],t_rtc *rtc,gmx_bool bStopCM);

/* Do a per group center of mass things */
void calc_vcm_grp(FILE *fp, int *la2ga, int start, int homenr, t_mdatoms *md,
                  rvec x[], rvec v[], t_vcm *vcm);

void do_stopcm_grp(FILE *fp, int *la2ga, int start, int homenr,
                   unsigned short *group_id,
                   rvec x[], rvec v[], t_vcm *vcm);

void check_cm_grp(FILE *fp, t_vcm *vcm, t_inputrec *ir, real Temp_Max);


#ifdef __cplusplus
}
#endif


#endif /* _vcm_h */
