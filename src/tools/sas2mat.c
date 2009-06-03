/*
 * $Id$
 * 
 *                This source code is part of
 * 
 *                 G   R   O   M   A   C   S
 * 
 *          GROningen MAchine for Chemical Simulations
 * 
 *                        VERSION 3.2.0
 * Written by David van der Spoel, Erik Lindahl, Berk Hess, and others.
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team,
 * check out http://www.gromacs.org for more information.

 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * If you want to redistribute modifications, please consider that
 * scientific software is very special. Version control is crucial -
 * bugs must be traceable. We will be happy to consider code for
 * inclusion in the official distribution, but derived work must not
 * be called official GROMACS. Details are found in the README & COPYING
 * files - if they are missing, get the official version at www.gromacs.org.
 * 
 * To help us fund GROMACS development, we humbly ask that you cite
 * the papers on the package - you can find them in the top README file.
 * 
 * For more info, check our website at http://www.gromacs.org
 * 
 * And Hey:
 * Green Red Orange Magenta Azure Cyan Skyblue
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>

#include "sysstuff.h"
#include "matio.h"
#include "copyrite.h"
#include "macros.h"
#include "statutil.h"
#include "smalloc.h"

int main(int argc,char *argv[])
{
  const char *desc[] = {
    "sas2mat converts matrix data in [IT]raw[it] format to X PixMap format,",
    "which can be digested by xpm2ps to make nice plots.",
    "These [IT]raw[it] data may be generated by g_rms, do_dssp or your",
    "own program.[PAR]",
    "The program prompts the user for some parameters:[PAR]",
    "[TT]Enter nres, res0, nframes, dt, t0, nlevels:[tt][PAR]",
    "In this context nres is the number of residues, res0 the starting residue",
    "dt is the time step, t0 is the starting time, nlevels is the number",
    "of levels for coloring. By default a greyscale colormap is generated."
  };
  static bool   bCol=FALSE;
  static char   *title="Area (nm^2)";
  static real   ssmin=-1,ssmax=-1,t0=0,dt=1;
  static int    nres=1,nframes=1,r0=0,nlevels=20,nskip=0;
  t_pargs pa[] = {
    { "-col",     FALSE,  etBOOL, &bCol,
      "The user is prompted for rgb lower and upper values" },
    { "-min",     FALSE,  etREAL, &ssmin,
      "Lower values for the data, calculated from the data by default" },
    { "-max",     FALSE,  etREAL, &ssmax,
      "Upper values for the data, see above" },
    { "-title",   FALSE,  etSTR,  &title,
      "Title for the graph" },
    { "-nlevel",  FALSE,  etINT,  &nlevels,
      "Number of levels in graph" },
    { "-nres",    FALSE,  etINT,  &nres,
      "Number of residues (Y-axis)" },
    { "-nframes", FALSE,  etINT,  &nframes,
      "Number of frames (Y-axis)" },
    { "-res0",    FALSE,  etINT,  &r0,
      "Number of first residue" },
    { "-nskip",   FALSE,  etINT,  &nskip,
      "Number of frames to skip after every frame" },
    { "-dt",      FALSE,  etREAL, &dt,
      "Time between time frames" },
    { "-t0",      FALSE,  etREAL, &t0,
      "Time of first time frame" }
  };
  
  FILE   *in,*out;
  int    i,j,k,ihi;
  double s;
  real   **ss,lo,hi,s1min,s1max;
  real   *resnr,*t;
  bool   bCheck=TRUE;
  t_rgb  rlo,rhi;
  t_filenm fnm[] = {
    { efOUT, "-f", "area", ffREAD },
    { efXPM, "-o", "sas",  ffWRITE }
  };
#define NFILE asize(fnm)

  /* If we want to read all frames nskip must be greater than zero */
  nskip += 1;

  CopyRight(stderr,argv[0]);
  
  parse_common_args(&argc,argv,PCA_BE_NICE,NFILE,fnm,asize(pa),pa,asize(desc),desc,
		    0,NULL);
  
  snew(ss,nres);
  snew(resnr,nres);
  snew(t,nframes);
  for(i=0; (i<nframes); i++) 
    t[i]=t0+i*dt;
  for(i=0; (i<nres); i++) {
    snew(ss[i],nframes);
  }
  in=ftp2FILE(efOUT,NFILE,fnm,"r");
  for(i=k=0; (i<nframes); i++) {
    for(j=0; (j<nres); j++) {
      fscanf(in,"%lf",&s);
      ss[j][k]=s;
    }
    if (!nskip || ((i % nskip) == 0))
      k++;
  }
  fclose(in);
  nframes=k;

  lo=10000;
  hi=0;
  for(j=0; (j<nres); j++) {
    /* Find lowest SAS value and subtract that from all occurrences */
    s1min=10000;
    s1max=0;
    for(i=0; (i<nframes); i++) {
      s1min=min(s1min,ss[j][i]);
      s1max=max(s1max,ss[j][i]);
    }
    printf("res %d: ssmin=%g, ssmax=%g, diff=%g\n",j,s1min,s1max,s1max-s1min);
    hi=max(hi,s1max);
    lo=min(lo,s1min);
  }
  printf("Lowest and Highest SAS value: %g %g\n",lo,hi);

  if (ssmin == -1)
    ssmin=lo;
  if (ssmax == -1)
    ssmax=hi;
  
  /*
    hi=ssmax-ssmin;
    for(j=0; (j<nres); j++) {
    for(i=0; (i<nframes); i++) 
    ss[j][i]-=ssmin;
    }
    */

  /* ihi=hi; */
  rhi.r=0,rhi.g=0,rhi.b=0;
  rlo.r=1,rlo.g=1,rlo.b=1;
  if (bCol) {
    printf("Color entries:\n""drlo glo blo rhi ghi bhi\n");
    scanf("%f%f%f%f%f%f",&rlo.r,&rlo.g,&rlo.b,&rhi.r,&rhi.g,&rhi.b);
  }
  /*
  write_mapfile(ftp2fn(efMAP,NFILE,fnm),&nlevels,rlo,rhi,ssmin,ssmax);
  */

  for(i=0;i<nres;i++)
    resnr[i]=i+1;
  out=ftp2FILE(efXPM,NFILE,fnm,"w");
  /*
  write_matrix(out,nres,nframes,resnr,t,ss,NULL,title,0,hi,nlevels);
  */
  write_xpm(out,0,"????","????","Time (ps)","Residue",
	    nres,nframes,resnr,t,ss,ssmin,ssmax,rlo,rhi,&nlevels);

  thanx(stderr);
  
  return 0;
}
