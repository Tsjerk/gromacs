#include <stdio.h>
#include <math.h>
#include "smalloc.h"
#include "typedefs.h"
#include "lrutil.h"
#include "macros.h"
#include "physics.h"
#include "nrnb.h"
#include "vec.h"
#include "pppm.h"

#define llim2 (-3)
#define ulim2  (3)

/* typedef for poisson solver */
typedef struct {
  int  nx,ny,nz;
  real ***ptr;
} t_PSgrid;

t_PSgrid *mk_PSgrid(int nx,int ny,int nz)
{
  t_PSgrid *ps;
  int      i,j;
  
  snew(ps,1);
  ps->nx=nx;
  ps->ny=ny;
  ps->nz=nz;
  snew(ps->ptr,nx);
  for(i=0; (i<nx); i++) {
    snew(ps->ptr[i],ny);
    for(j=0; (j<ny); j++)
      snew(ps->ptr[i][j],nz);
  }
  return ps;
}

void unpack_PSgrid(t_PSgrid *grid,int *nx,int *ny,int *nz,real ****ptr)
{
  *nx  = grid->nx;
  *ny  = grid->ny;
  *nz  = grid->nz;
  *ptr = grid->ptr;
}

void copy_PSgrid(t_PSgrid *dest,t_PSgrid *src)
{
  int  i,j,k;
  int  nx,ny,nz;
  real ***src_ptr,***dst_ptr;
  
  unpack_PSgrid(dest,&nx,&ny,&nz,&dst_ptr);
  unpack_PSgrid(src,&nx,&ny,&nz,&src_ptr);
  
  for(i=0; (i<nx); i++)
    for(j=0; (j<ny); j++)
      for(k=0; (k<nz); k++)
	dst_ptr[i][j][k] = src_ptr[i][j][k];
}

void clear_PSgrid(t_PSgrid *grid)
{
  int  i,j,k;
  int  nx,ny,nz;
  real ***ptr;
  
  unpack_PSgrid(grid,&nx,&ny,&nz,&ptr);
  
  for(i=0; (i<nx); i++)
    for(j=0; (j<ny); j++)
      for(k=0; (k<nz); k++)
	ptr[i][j][k] = 0.0;
}

static void calc_nxyz(int nx,int ny,int nz,
		      int **nnx,int **nny,int **nnz)
{
  int i;
  
  snew(*nnx,3*nx);
  snew(*nny,3*ny);
  snew(*nnz,3*nz);
  for(i=0; (i<3*nx); i++)
    (*nnx)[i] = i % nx;
  for(i=0; (i<3*ny); i++)
    (*nny)[i] = i % ny;
  for(i=0; (i<3*nz); i++)
    (*nnz)[i] = i % nz;
}
	
static void calc_invh(rvec box,int nx,int ny,int nz,rvec invh)
{
  invh[XX] = nx/box[XX];
  invh[YY] = ny/box[YY];
  invh[ZZ] = nz/box[ZZ];
}

real ps_gather_inner(int JCXYZ[],real WXYZ[],int ixw[],int iyw[],int izw[],
		     real c1x,real c1y,real c1z,real c2x,real c2y,real c2z,
		     real qi,rvec f,real ***ptr)
{
  real pi,fX,fY,fZ,weight;
  int  jxyz,m,jcx,jcy,jcz;
  int  jcx0,jcy0,jcz0;
  
  pi = 0.0;
  fX = 0.0;
  fY = 0.0;
  fZ = 0.0;
  
  /* Now loop over 27 surrounding vectors */      
  for(jxyz=m=0; (jxyz < 27); jxyz++,m+=3) {
    jcx    = JCXYZ[m];
    jcy    = JCXYZ[m+1];
    jcz    = JCXYZ[m+2];
    weight = WXYZ[jxyz];
    
    jcx0   = ixw[jcx];
    jcy0   = iyw[jcy];
    jcz0   = izw[jcz];

    /* Electrostatic Potential ! */
    pi += weight * ptr[jcx0][jcy0][jcz0];
  
    /* Forces */
    fX += weight * ((c1x*(ptr[ixw[jcx-1]] [jcy0]       [jcz0] - 
			  ptr[ixw[jcx+1]] [jcy0]       [jcz0] )) +
		    (c2x*(ptr[ixw[jcx-2]] [jcy0]       [jcz0] - 
			  ptr[ixw[jcx+2]] [jcy0]       [jcz0] )));
    fY += weight * ((c1y*(ptr[jcx0]       [iyw[jcy-1]] [jcz0] -
			  ptr[jcx0]       [iyw[jcy+1]] [jcz0] ))  +
		    (c2y*(ptr[jcx0]       [iyw[jcy-2]] [jcz0] -
			  ptr[jcx0]       [iyw[jcy+2]] [jcz0] )));
    fZ += weight * ((c1z*(ptr[jcx0]       [jcy0]       [izw[jcz-1]] -
			  ptr[jcx0]       [jcy0]       [izw[jcz+1]] ))  +
		    (c2z*(ptr[jcx0]       [jcy0]       [izw[jcz-2]] -
			  ptr[jcx0]       [jcy0]       [izw[jcz+2]] )));
  }
  f[XX] += qi*fX;
  f[YY] += qi*fY;
  f[ZZ] += qi*fZ;
  
  return pi;
}

real ps_gather_f(FILE *log,bool bVerbose,
		 int natoms,rvec x[],rvec f[],real charge[],rvec box,
		 real pot[],t_PSgrid *grid,rvec beta,t_nrnb *nrnb)
{
  static bool bFirst=TRUE;
  static int  *nnx,*nny,*nnz;
  static int  JCXYZ[81];
  int    i,m;
  real   energy;
  real   qi,pi;
  ivec   ixyz;
  rvec   invh,c1,c2;
  real   WXYZ[27];
  real   c1x,c1y,c1z,c2x,c2y,c2z;
  int    ixw[7],iyw[7],izw[7];
  int    ll;
  int    nx,ny,nz;
  real   ***ptr;
  
  unpack_PSgrid(grid,&nx,&ny,&nz,&ptr);
  
  calc_invh(box,nx,ny,nz,invh);
  
  for(m=0; (m<DIM); m++) {
    c1[m] = (beta[m]/2.0)*invh[m];
    c2[m] = ((1.0-beta[m])/4.0)*invh[m];
  }
  c1x = c1[XX];
  c1y = c1[YY];
  c1z = c1[ZZ];
  c2x = c2[XX];
  c2y = c2[YY];
  c2z = c2[ZZ];

  if (bFirst) {
    fprintf(log,"Gathering Forces using Triangle Shaped on %dx%dx%d grid\n",
	    nx,ny,nz);
    fprintf(log,"beta = %10g,%10g,%10g\n",beta[XX],beta[YY],beta[ZZ]);
    fprintf(log,"c1   = %10g,%10g,%10g\n",c1[XX],c1[YY],c1[ZZ]);
    fprintf(log,"c2   = %10g,%10g,%10g\n",c2[XX],c2[YY],c2[ZZ]);
    fprintf(log,"invh = %10g,%10g,%10g\n",invh[XX],invh[YY],invh[ZZ]);

    calc_nxyz(nx,ny,nz,&nnx,&nny,&nnz);

    for(i=0; (i<27); i++) {
      JCXYZ[3*i]   = 2 + (i/9);
      JCXYZ[3*i+1] = 2 + (i/3) % 3;
      JCXYZ[3*i+2] = 2 + (i % 3); 
    }
    
    bFirst = FALSE;
  }

  energy=0.0;  	  
  for(i=0; (i<natoms); i++) {
    /* Each charge is spread over the nearest 27 grid cells,
     * thus we loop over -1..1 in X,Y and Z direction
     * We apply the TSC (triangle shaped charge)
     * see Luty et. al, JCP 103 (1995) 3014
     */
     
    calc_weights(nx,ny,nz,x[i],box,invh,ixyz,WXYZ);

    for(ll=llim2; (ll<=ulim2); ll++) {
      ixw[ll-llim2] = nnx[ixyz[XX]+ll+nx];
      iyw[ll-llim2] = nny[ixyz[YY]+ll+ny];
      izw[ll-llim2] = nnz[ixyz[ZZ]+ll+nz];
    }
    
    qi      = charge[i];
    pi      = ps_gather_inner(JCXYZ,WXYZ,ixw,iyw,izw,
			      c1x,c1y,c1z,c2x,c2y,c2z,
			      qi,f[i],ptr);
    
    energy += pi*qi;
    pot[i]  = pi;
  }
  
  inc_nrnb(nrnb,eNR_GATHERF,27*natoms);
  inc_nrnb(nrnb,eNR_WEIGHTS,3*natoms);
  
  return energy*0.5;
}



static void spread_q_poisson(FILE *log,bool bVerbose,
			     int natoms,rvec x[],real charge[],rvec box,
			     real r1,real rc,t_PSgrid *grid,t_nrnb *nrnb)
{
  static bool bFirst = TRUE;
  static int  *nnx,*nny,*nnz;
  rvec   invh;
  real   qi,qt,qwt;
  rvec   gridpoint,dx;
  real   WXYZ[27],bhh,r,half=0.5;
  ivec   ixyz;
  int    i,iX,iY,iZ,index,ttt,m;
  int    jx,jy,jz,jcx,jcy,jcz;
  int    nxyz,ncellsx,ncellsy,ncellsz;
  int    nx,ny,nz,la1,la2,la12;
  int    xmin,xmax,ymin,ymax,zmin,zmax;
  real   ***rho;
  
  unpack_PSgrid(grid,&nx,&ny,&nz,&rho);
  
  calc_invh(box,nx,ny,nz,invh);

  if (bFirst) {
    fprintf(log,"Spreading Charges using spread function on %dx%dx%d grid\n",
	    nx,ny,nz);
    fprintf(log,"invh = %10g,%10g,%10g\n",invh[XX],invh[YY],invh[ZZ]);
  
    calc_nxyz(nx,ny,nz,&nnx,&nny,&nnz);
    
    bFirst = FALSE;
  }
  ncellsx=(rc/box[XX])*nx;
  ncellsy=(rc/box[YY])*ny;
  ncellsz=(rc/box[ZZ])*nz;
  
  /* Has to be parallellized too! */
  for(i=0; (i<natoms); i++) {
    qi=charge[i];
    
    if (qi != 0.0) {
      /* Determine position of particle in box */
      for(m=0; (m<DIM); m++) {
	/* Put particle in the box... */  
	ttt = x[i][m]*invh[m];
	bhh = box[m]*invh[m];
	if (ttt < 0)
	  ttt += bhh;
	else if (ttt >= bhh)
	  ttt -= bhh;
      
	/* Calculate nearest grid point, Round */
	ixyz[m]    = ttt+half;
      }
      xmin = ixyz[XX] + nx - ncellsx;
      ymin = ixyz[YY] + ny - ncellsy;
      zmin = ixyz[ZZ] + nz - ncellsz;
      xmax = xmin+min(nx,2*ncellsx+1);
      ymax = ymin+min(ny,2*ncellsy+1);
      zmax = zmin+min(nz,2*ncellsz+1);
      
      for(jx=xmin; (jx<=xmax); jx++) {
	jcx = nnx[jx];
	gridpoint[XX] = (jcx*box[XX])/nx;
	for(jy=ymin; (jy<=ymax); jy++) {
	  jcy = nny[jy];
	  gridpoint[YY] = (jcy*box[YY])/ny;
	  for(jz=zmin; (jz<=zmax); jz++) {
	    jcz = nnz[jz]; 
	    gridpoint[ZZ] = (jcz*box[ZZ])/nz;
	    
	    rvec_sub(gridpoint,x[i],dx);
	    r = norm(dx);
	    rho[jcx][jcy][jcz] += qi*spreadfunction(r1,rc,r);
	  }
	}
      }
    }
  }
}

void solve_poisson(FILE *log,t_PSgrid *pot,t_PSgrid *rho,
		   bool bVerbose,t_nrnb *nrnb,int maxnit)
{
  static bool bFirst = TRUE;
  static int  *nnx,*nny,*nnz;
  const  real sixth=1.0/6.0,tol=0.001,eps0=4*M_PI*ONE_4PI_EPS0;
  real   deviation,val_ijk,epsrho;
  int    nit;
  int    i,j,k,i_1,i1,j_1,j1,k_1,k1;
  int    nx,ny,nz;
  real   ***pot_ptr,***rho_ptr;
  
  unpack_PSgrid(pot,&nx,&ny,&nz,&pot_ptr);
  unpack_PSgrid(rho,&nx,&ny,&nz,&rho_ptr);

  if (bFirst) {
    fprintf(log,"Solving Poisson Equation on %dx%dx%d grid\n",
	    nx,ny,nz);
  
    calc_nxyz(nx,ny,nz,&nnx,&nny,&nnz);
    
    bFirst = FALSE;
  }

  /* Solve by simple averaging */
  nit = 0;
  do {
    deviation = 0.0;
    for(i=0; (i<nx); i++) {
      /* Indices left and right in the grid */
      i_1 = nnx[i-1+nx];
      i1  = nnx[i+1+nx];
      for(j=0; (j<ny); j++) {
	/* Indices fore and back in the grid */
	j_1 = nny[j-1+ny];
	j1  = nny[j+1+ny];
	for(k=0; (k<nz); k++) {
	  /* Indices fore and back in the grid */
	  k_1 = nnz[k-1+nz];
	  k1  = nnz[k+1+nz];
	  /* Get the new value by averaging surrounding grid points */
	  epsrho  = eps0*rho_ptr[i][j][k];
	  val_ijk = pot_ptr[i][j][k] = sixth*
	    (epsrho - pot_ptr[i_1][j][k] + pot_ptr[i1][j][k] +
	     pot_ptr[i][j_1][k] + pot_ptr[i][j1][k] +
	     pot_ptr[i][j][k_1] + pot_ptr[i][j][k1]);
	  deviation += sqr(val_ijk - epsrho);
	}
      }
    }
    if (bVerbose)
      fprintf(stderr,"\rnit: %5d  dev: %8.3f\n",nit,sqrt(deviation));
    
    nit ++;
  } while ((nit < maxnit) && (deviation > tol));
  if (bVerbose)
    fprintf(stderr,"\n");
}

real do_poisson(FILE *log,       bool bVerbose,
		t_inputrec *ir,  int natoms,
		rvec x[],        rvec f[],
		real charge[],   rvec box,
		real phi[],      t_commrec *cr,
		t_nrnb *nrnb)
{
  static  bool bFirst  = TRUE;
  static  bool bSecond = TRUE;
  static  t_PSgrid *pot,*rho;
  static  int       niter;
  static  real      r1,rc;
  static  rvec      beta;
  
  const     real tol = 1e-5;
  int       i,m;
  real      ctot;
  real      aver,tot,ener;
  ivec      grids;
  rvec      spacing;
  
  ener = 0.0;
  
  if (bFirst) {
    niter = ir->niter;

    fprintf(log,"Will use Poisson Solver for long-range electrostatics\n");
    fprintf(log,"Grid size is %d x %d x %d\n",ir->nkx,ir->nky,ir->nkz);

    if ((ir->nkx < 4) || (ir->nky < 4) || (ir->nkz < 4)) 
      fatal_error(0,"Grid must be at least 4 points in all directions");
    
    pot = mk_PSgrid(ir->nkx,ir->nky,ir->nkz);
    rho = mk_PSgrid(ir->nkx,ir->nky,ir->nkz);
    
    r1 = ir->rshort;
    rc = ir->rlong;
    for(m=0; (m<DIM); m++)
      beta[m] = 4.0/3.0;
      
    bFirst = FALSE;
  }
  else {
    /* Make the grid empty */
    clear_PSgrid(rho);
    spread_q_poisson(log,bVerbose,natoms,x,charge,box,r1,rc,rho,nrnb);
    
    if (bSecond)
      copy_PSgrid(pot,rho);
    
    /* Second step: solving the poisson equation in real space */
    solve_poisson(log,rho,pot,bVerbose,nrnb,niter);
    
    /* Third and last step: gather the forces, energies and potential
     * from the grid.
     */
    ener = ps_gather_f(log,bVerbose,natoms,x,f,charge,box,phi,pot,beta,nrnb);
    
    bSecond = FALSE;
  }
  
  return ener;
}

