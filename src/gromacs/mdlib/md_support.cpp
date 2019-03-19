/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2016,2017,2018,2019, by the GROMACS development team, led by
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

#include "gmxpre.h"

#include "md_support.h"

#include <climits>
#include <cmath>

#include <algorithm>

#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/partition.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/mdrun.h"
#include "gromacs/mdlib/sim_util.h"
#include "gromacs/mdlib/simulationsignal.h"
#include "gromacs/mdlib/tgroup.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdlib/vcm.h"
#include "gromacs/mdrunutility/accumulateglobals.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/df_history.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/energyhistory.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pulling/pull.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/snprintf.h"

// TODO move this to multi-sim module
bool multisim_int_all_are_equal(const gmx_multisim_t *ms,
                                int64_t               value)
{
    bool         allValuesAreEqual = true;
    int64_t     *buf;

    GMX_RELEASE_ASSERT(ms, "Invalid use of multi-simulation pointer");

    snew(buf, ms->nsim);
    /* send our value to all other master ranks, receive all of theirs */
    buf[ms->sim] = value;
    gmx_sumli_sim(ms->nsim, buf, ms);

    for (int s = 0; s < ms->nsim; s++)
    {
        if (buf[s] != value)
        {
            allValuesAreEqual = false;
            break;
        }
    }

    sfree(buf);

    return allValuesAreEqual;
}

int multisim_min(const gmx_multisim_t *ms, int nmin, int n)
{
    int     *buf;
    gmx_bool bPos, bEqual;
    int      s, d;

    snew(buf, ms->nsim);
    buf[ms->sim] = n;
    gmx_sumi_sim(ms->nsim, buf, ms);
    bPos   = TRUE;
    bEqual = TRUE;
    for (s = 0; s < ms->nsim; s++)
    {
        bPos   = bPos   && (buf[s] > 0);
        bEqual = bEqual && (buf[s] == buf[0]);
    }
    if (bPos)
    {
        if (bEqual)
        {
            nmin = std::min(nmin, buf[0]);
        }
        else
        {
            /* Find the least common multiple */
            for (d = 2; d < nmin; d++)
            {
                s = 0;
                while (s < ms->nsim && d % buf[s] == 0)
                {
                    s++;
                }
                if (s == ms->nsim)
                {
                    /* We found the LCM and it is less than nmin */
                    nmin = d;
                    break;
                }
            }
        }
    }
    sfree(buf);

    return nmin;
}

/* TODO Specialize this routine into init-time and loop-time versions?
   e.g. bReadEkin is only true when restoring from checkpoint */
void compute_globals(FILE *fplog, gmx_global_stat *gstat, t_commrec *cr, t_inputrec *ir,
                     t_forcerec *fr, gmx_ekindata_t *ekind,
                     rvec *x, rvec *v, matrix box, real vdwLambda, t_mdatoms *mdatoms,
                     t_nrnb *nrnb, t_vcm *vcm, gmx_wallcycle_t wcycle,
                     gmx_enerdata_t *enerd, tensor force_vir, tensor shake_vir, tensor total_vir,
                     tensor pres, rvec mu_tot, gmx::Constraints *constr,
                     gmx::SimulationSignaller *signalCoordinator,
                     matrix lastbox,
                     gmx::AccumulateGlobals *accumulateGlobals,
                     int *totalNumberOfBondedInteractions,
                     gmx_bool *bSumEkinhOld, int flags)
{
    tensor   corr_vir, corr_pres;
    gmx_bool bEner, bPres, bTemp;
    gmx_bool bStopCM, bGStat,
             bReadEkin, bEkinAveVel, bScaleEkin, bConstrain;
    gmx_bool bCheckNumberOfBondedInteractions;
    real     prescorr, enercorr, dvdlcorr, dvdl_ekin;

    /* translate CGLO flags to gmx_booleans */
    bStopCM                          = ((flags & CGLO_STOPCM) != 0);
    bGStat                           = ((flags & CGLO_GSTAT) != 0);
    bReadEkin                        = ((flags & CGLO_READEKIN) != 0);
    bScaleEkin                       = ((flags & CGLO_SCALEEKIN) != 0);
    bEner                            = ((flags & CGLO_ENERGY) != 0);
    bTemp                            = ((flags & CGLO_TEMPERATURE) != 0);
    bPres                            = ((flags & CGLO_PRESSURE) != 0);
    bConstrain                       = ((flags & CGLO_CONSTRAINT) != 0);
    bCheckNumberOfBondedInteractions = ((flags & CGLO_CHECK_NUMBER_OF_BONDED_INTERACTIONS) != 0);

    /* we calculate a full state kinetic energy either with full-step velocity verlet
       or half step where we need the pressure */

    bEkinAveVel = (ir->eI == eiVV || (ir->eI == eiVVAK && bPres) || bReadEkin);

    /* in initalization, it sums the shake virial in vv, and to
       sums ekinh_old in leapfrog (or if we are calculating ekinh_old) for other reasons */

    /* ########## Kinetic energy  ############## */

    if (bTemp)
    {
        /* Non-equilibrium MD: this is parallellized, but only does communication
         * when there really is NEMD.
         */

        if (PAR(cr) && (ekind->bNEMD))
        {
            accumulate_u(cr, &(ir->opts), ekind);
        }
        if (!bReadEkin)
        {
            calc_ke_part(
                    x, v, box,
                    &(ir->opts), mdatoms, ekind, nrnb, bEkinAveVel);
        }
    }

    /* Calculate center of mass velocity if necessary, also parallellized */
    if (bStopCM)
    {
        calc_vcm_grp(0, mdatoms->homenr, mdatoms,
                     x, v, vcm);
    }

    if (bTemp || bStopCM || bPres || bEner || bConstrain || bCheckNumberOfBondedInteractions)
    {
        if (!bGStat)
        {
            /* We will not sum ekinh_old,
             * so signal that we still have to do it.
             */
            *bSumEkinhOld = TRUE;

        }
        else
        {
            gmx::ArrayRef<real> signalBuffer = signalCoordinator->getCommunicationBuffer();
            if (PAR(cr))
            {
                wallcycle_start(wcycle, ewcMoveE);
                global_stat(gstat, cr, enerd, force_vir, shake_vir, mu_tot,
                            ir, ekind, constr, bStopCM ? vcm : nullptr,
                            signalBuffer.size(), signalBuffer.data(),
                            accumulateGlobals->getReductionView(),
                            totalNumberOfBondedInteractions,
                            *bSumEkinhOld, flags);
                accumulateGlobals->notifyClientsAfterCommunication();
                wallcycle_stop(wcycle, ewcMoveE);
            }
            signalCoordinator->finalizeSignals();
            *bSumEkinhOld = FALSE;
        }
    }

    /* Do center of mass motion removal */
    if (bStopCM)
    {
        check_cm_grp(fplog, vcm, ir, 1);
        /* At initialization, do not pass x with acceleration-correction mode
         * to avoid (incorrect) correction of the initial coordinates.
         */
        rvec *xPtr = nullptr;
        if (vcm->mode == ecmANGULAR || (vcm->mode == ecmLINEAR_ACCELERATION_CORRECTION && !(flags & CGLO_INITIALIZATION)))
        {
            xPtr = x;
        }
        do_stopcm_grp(*mdatoms,
                      xPtr, v, *vcm);
        inc_nrnb(nrnb, eNR_STOPCM, mdatoms->homenr);
    }

    if (bEner)
    {
        /* Calculate the amplitude of the cosine velocity profile */
        ekind->cosacc.vcos = ekind->cosacc.mvcos/mdatoms->tmass;
    }

    if (bTemp)
    {
        /* Sum the kinetic energies of the groups & calc temp */
        /* compute full step kinetic energies if vv, or if vv-avek and we are computing the pressure with inputrecNptTrotter */
        /* three maincase:  VV with AveVel (md-vv), vv with AveEkin (md-vv-avek), leap with AveEkin (md).
           Leap with AveVel is not supported; it's not clear that it will actually work.
           bEkinAveVel: If TRUE, we simply multiply ekin by ekinscale to get a full step kinetic energy.
           If FALSE, we average ekinh_old and ekinh*ekinscale_nhc to get an averaged half step kinetic energy.
         */
        enerd->term[F_TEMP] = sum_ekin(&(ir->opts), ekind, &dvdl_ekin,
                                       bEkinAveVel, bScaleEkin);
        enerd->dvdl_lin[efptMASS] = static_cast<double>(dvdl_ekin);

        enerd->term[F_EKIN] = trace(ekind->ekin);
    }

    /* ##########  Long range energy information ###### */

    if (bEner || bPres || bConstrain)
    {
        calc_dispcorr(ir, fr, lastbox, vdwLambda,
                      corr_pres, corr_vir, &prescorr, &enercorr, &dvdlcorr);
    }

    if (bEner)
    {
        enerd->term[F_DISPCORR]  = enercorr;
        enerd->term[F_EPOT]     += enercorr;
        enerd->term[F_DVDL_VDW] += dvdlcorr;
    }

    /* ########## Now pressure ############## */
    if (bPres || bConstrain)
    {

        m_add(force_vir, shake_vir, total_vir);

        /* Calculate pressure and apply LR correction if PPPM is used.
         * Use the box from last timestep since we already called update().
         */

        enerd->term[F_PRES] = calc_pres(fr->ePBC, ir->nwall, lastbox, ekind->ekin, total_vir, pres);

        /* Calculate long range corrections to pressure and energy */
        /* this adds to enerd->term[F_PRES] and enerd->term[F_ETOT],
           and computes enerd->term[F_DISPCORR].  Also modifies the
           total_vir and pres tesors */

        m_add(total_vir, corr_vir, total_vir);
        m_add(pres, corr_pres, pres);
        enerd->term[F_PDISPCORR] = prescorr;
        enerd->term[F_PRES]     += prescorr;
    }
}

/* check whether an 'nst'-style parameter p is a multiple of nst, and
   set it to be one if not, with a warning. */
static void check_nst_param(const gmx::MDLogger &mdlog,
                            const char *desc_nst, int nst,
                            const char *desc_p, int *p)
{
    if (*p > 0 && *p % nst != 0)
    {
        /* Round up to the next multiple of nst */
        *p = ((*p)/nst + 1)*nst;
        GMX_LOG(mdlog.warning).asParagraph().appendTextFormatted(
                "NOTE: %s changes %s to %d", desc_nst, desc_p, *p);
    }
}

void setCurrentLambdasRerun(int64_t step, const t_lambda *fepvals,
                            const t_trxframe *rerun_fr, const double *lam0,
                            t_state *globalState)
{
    GMX_RELEASE_ASSERT(globalState != nullptr, "setCurrentLambdasGlobalRerun should be called with a valid state object");

    if (rerun_fr->bLambda)
    {
        if (fepvals->delta_lambda == 0)
        {
            globalState->lambda[efptFEP] = rerun_fr->lambda;
        }
        else
        {
            /* find out between which two value of lambda we should be */
            real frac      = step*fepvals->delta_lambda;
            int  fep_state = static_cast<int>(std::floor(frac*fepvals->n_lambda));
            /* interpolate between this state and the next */
            /* this assumes that the initial lambda corresponds to lambda==0, which is verified in grompp */
            frac           = frac*fepvals->n_lambda - fep_state;
            for (int i = 0; i < efptNR; i++)
            {
                globalState->lambda[i] = lam0[i] + (fepvals->all_lambda[i][fep_state]) +
                    frac*(fepvals->all_lambda[i][fep_state+1] - fepvals->all_lambda[i][fep_state]);
            }
        }
    }
    else if (rerun_fr->bFepState)
    {
        globalState->fep_state = rerun_fr->fep_state;
        for (int i = 0; i < efptNR; i++)
        {
            globalState->lambda[i] = fepvals->all_lambda[i][globalState->fep_state];
        }
    }
}

void setCurrentLambdasLocal(int64_t step, const t_lambda *fepvals,
                            const double *lam0, t_state *state)
/* find the current lambdas.  If rerunning, we either read in a state, or a lambda value,
   requiring different logic. */
{
    if (fepvals->delta_lambda != 0)
    {
        /* find out between which two value of lambda we should be */
        real frac = step*fepvals->delta_lambda;
        if (fepvals->n_lambda > 0)
        {
            int fep_state = static_cast<int>(std::floor(frac*fepvals->n_lambda));
            /* interpolate between this state and the next */
            /* this assumes that the initial lambda corresponds to lambda==0, which is verified in grompp */
            frac          = frac*fepvals->n_lambda - fep_state;
            for (int i = 0; i < efptNR; i++)
            {
                state->lambda[i] = lam0[i] + (fepvals->all_lambda[i][fep_state]) +
                    frac*(fepvals->all_lambda[i][fep_state + 1] - fepvals->all_lambda[i][fep_state]);
            }
        }
        else
        {
            for (int i = 0; i < efptNR; i++)
            {
                state->lambda[i] = lam0[i] + frac;
            }
        }
    }
    else
    {
        /* if < 0, fep_state was never defined, and we should not set lambda from the state */
        if (state->fep_state > -1)
        {
            for (int i = 0; i < efptNR; i++)
            {
                state->lambda[i] = fepvals->all_lambda[i][state->fep_state];
            }
        }
    }
}

static void min_zero(int *n, int i)
{
    if (i > 0 && (*n == 0 || i < *n))
    {
        *n = i;
    }
}

static int lcd4(int i1, int i2, int i3, int i4)
{
    int nst;

    nst = 0;
    min_zero(&nst, i1);
    min_zero(&nst, i2);
    min_zero(&nst, i3);
    min_zero(&nst, i4);
    if (nst == 0)
    {
        gmx_incons("All 4 inputs for determining nstglobalcomm are <= 0");
    }

    while (nst > 1 && ((i1 > 0 && i1 % nst != 0)  ||
                       (i2 > 0 && i2 % nst != 0)  ||
                       (i3 > 0 && i3 % nst != 0)  ||
                       (i4 > 0 && i4 % nst != 0)))
    {
        nst--;
    }

    return nst;
}

int check_nstglobalcomm(const gmx::MDLogger &mdlog, int nstglobalcomm, t_inputrec *ir, const t_commrec * cr)
{
    if (!EI_DYNAMICS(ir->eI))
    {
        nstglobalcomm = 1;
    }

    if (nstglobalcomm == -1)
    {
        // Set up the default behaviour
        if (!(ir->nstcalcenergy > 0 ||
              ir->nstlist > 0 ||
              ir->etc != etcNO ||
              ir->epc != epcNO))
        {
            /* The user didn't choose the period for anything
               important, so we just make sure we can send signals and
               write output suitably. */
            nstglobalcomm = 10;
            if (ir->nstenergy > 0 && ir->nstenergy < nstglobalcomm)
            {
                nstglobalcomm = ir->nstenergy;
            }
        }
        else
        {
            /* The user has made a choice (perhaps implicitly), so we
             * ensure that we do timely intra-simulation communication
             * for (possibly) each of the four parts that care.
             *
             * TODO Does the Verlet scheme (+ DD) need any
             * communication at nstlist steps? Is the use of nstlist
             * here a leftover of the twin-range scheme? Can we remove
             * nstlist when we remove the group scheme?
             */
            nstglobalcomm = lcd4(ir->nstcalcenergy,
                                 ir->nstlist,
                                 ir->etc != etcNO ? ir->nsttcouple : 0,
                                 ir->epc != epcNO ? ir->nstpcouple : 0);
        }
    }
    else
    {
        // Check that the user's choice of mdrun -gcom will work
        if (ir->nstlist > 0 &&
            nstglobalcomm > ir->nstlist && nstglobalcomm % ir->nstlist != 0)
        {
            nstglobalcomm = (nstglobalcomm / ir->nstlist)*ir->nstlist;
            GMX_LOG(mdlog.warning).asParagraph().appendTextFormatted(
                    "WARNING: nstglobalcomm is larger than nstlist, but not a multiple, setting it to %d",
                    nstglobalcomm);
        }
        if (ir->nstcalcenergy > 0)
        {
            check_nst_param(mdlog, "-gcom", nstglobalcomm,
                            "nstcalcenergy", &ir->nstcalcenergy);
        }
        if (ir->etc != etcNO && ir->nsttcouple > 0)
        {
            check_nst_param(mdlog, "-gcom", nstglobalcomm,
                            "nsttcouple", &ir->nsttcouple);
        }
        if (ir->epc != epcNO && ir->nstpcouple > 0)
        {
            check_nst_param(mdlog, "-gcom", nstglobalcomm,
                            "nstpcouple", &ir->nstpcouple);
        }

        check_nst_param(mdlog, "-gcom", nstglobalcomm,
                        "nstenergy", &ir->nstenergy);

        check_nst_param(mdlog, "-gcom", nstglobalcomm,
                        "nstlog", &ir->nstlog);
    }

    if (ir->comm_mode != ecmNO && ir->nstcomm < nstglobalcomm)
    {
        GMX_LOG(mdlog.warning).asParagraph().appendTextFormatted(
                "WARNING: Changing nstcomm from %d to %d",
                ir->nstcomm, nstglobalcomm);
        ir->nstcomm = nstglobalcomm;
    }

    if (cr->nnodes > 1)
    {
        GMX_LOG(mdlog.info).appendTextFormatted(
                "Intra-simulation communication will occur every %d steps.\n", nstglobalcomm);
    }
    return nstglobalcomm;

}

void rerun_parallel_comm(t_commrec *cr, t_trxframe *fr,
                         gmx_bool *bLastStep)
{
    rvec    *xp, *vp;

    if (MASTER(cr) && *bLastStep)
    {
        fr->natoms = -1;
    }
    xp = fr->x;
    vp = fr->v;
    gmx_bcast(sizeof(*fr), fr, cr);
    fr->x = xp;
    fr->v = vp;

    *bLastStep = (fr->natoms < 0);

}

// TODO Most of this logic seems to belong in the respective modules
void set_state_entries(t_state *state, const t_inputrec *ir)
{
    /* The entries in the state in the tpx file might not correspond
     * with what is needed, so we correct this here.
     */
    state->flags = 0;
    if (ir->efep != efepNO || ir->bExpanded)
    {
        state->flags |= (1<<estLAMBDA);
        state->flags |= (1<<estFEPSTATE);
    }
    state->flags |= (1<<estX);
    GMX_RELEASE_ASSERT(state->x.size() == state->natoms, "We should start a run with an initialized state->x");
    if (EI_DYNAMICS(ir->eI))
    {
        state->flags |= (1<<estV);
    }

    state->nnhpres = 0;
    if (ir->ePBC != epbcNONE)
    {
        state->flags |= (1<<estBOX);
        if (inputrecPreserveShape(ir))
        {
            state->flags |= (1<<estBOX_REL);
        }
        if ((ir->epc == epcPARRINELLORAHMAN) || (ir->epc == epcMTTK))
        {
            state->flags |= (1<<estBOXV);
            state->flags |= (1<<estPRES_PREV);
        }
        if (inputrecNptTrotter(ir) || (inputrecNphTrotter(ir)))
        {
            state->nnhpres = 1;
            state->flags  |= (1<<estNHPRES_XI);
            state->flags  |= (1<<estNHPRES_VXI);
            state->flags  |= (1<<estSVIR_PREV);
            state->flags  |= (1<<estFVIR_PREV);
            state->flags  |= (1<<estVETA);
            state->flags  |= (1<<estVOL0);
        }
        if (ir->epc == epcBERENDSEN)
        {
            state->flags  |= (1<<estBAROS_INT);
        }
    }

    if (ir->etc == etcNOSEHOOVER)
    {
        state->flags |= (1<<estNH_XI);
        state->flags |= (1<<estNH_VXI);
    }

    if (ir->etc == etcVRESCALE || ir->etc == etcBERENDSEN)
    {
        state->flags |= (1<<estTHERM_INT);
    }

    init_gtc_state(state, state->ngtc, state->nnhpres, ir->opts.nhchainlength); /* allocate the space for nose-hoover chains */
    init_ekinstate(&state->ekinstate, ir);

    if (ir->bExpanded)
    {
        snew(state->dfhist, 1);
        init_df_history(state->dfhist, ir->fepvals->n_lambda);
    }

    if (ir->pull && ir->pull->bSetPbcRefToPrevStepCOM)
    {
        state->flags |= (1<<estPULLCOMPREVSTEP);
    }
}

namespace gmx
{
ComputeGlobalsElement::ComputeGlobalsElement(
        std::shared_ptr<MicroState> &microState,
        gmx_enerdata_t              *enerd,
        tensor                       force_vir,
        tensor                       shake_vir,
        tensor                       total_vir,
        tensor                       pres,
        rvec                         mu_tot,
        FILE                        *fplog,
        const MDLogger              &mdlog,
        t_commrec                   *cr,
        t_inputrec                  *inputrec,
        t_mdatoms                   *mdatoms,
        t_nrnb                      *nrnb,
        t_forcerec                  *fr,
        gmx_wallcycle_t              wcycle,
        gmx_mtop_t                  *global_top,
        gmx_localtop_t              *top,
        gmx_ekindata_t              *ekind,
        Constraints                 *constr,
        t_vcm                       *vcm,
        int                          globalCommunicationInterval) :
    doStopCM_(inputrec->comm_mode != ecmNO),
    nstcomm_(inputrec->nstcomm),
    needGlobalReduction_(false),
    needEnergyReduction_(false),
    isVV_(EI_VV(inputrec->eI)),
    isLF_(EI_MD(inputrec->eI) && !EI_VV(inputrec->eI)),
    totalNumberOfBondedInteractions_(0),
    shouldCheckNumberOfBondedInteractions_(false),
    microState_(microState),
    fplog_(fplog),
    mdlog_(mdlog),
    cr_(cr),
    inputrec_(inputrec),
    top_global_(global_top),
    mdatoms_(mdatoms),
    enerd_(enerd),
    force_vir_(force_vir),
    shake_vir_(shake_vir),
    total_vir_(total_vir),
    pres_(pres),
    mu_tot_(mu_tot),
    ekind_(ekind),
    constr_(constr),
    nrnb_(nrnb),
    wcycle_(wcycle),
    fr_(fr),
    vcm_(vcm),
    top_(top),
    signals_()
{
    gstat_         = global_stat_init(inputrec_);
    nstglobalcomm_ = check_nstglobalcomm(mdlog, globalCommunicationInterval, inputrec, cr);
}

ComputeGlobalsElement::~ComputeGlobalsElement()
{
    global_stat_destroy(gstat_);
}

ElementFunctionTypePtr ComputeGlobalsElement::registerSetup()
{
    return std::make_unique<ElementFunctionType>(
            std::bind(&ComputeGlobalsElement::setup, this));
}

ElementFunctionTypePtr ComputeGlobalsElement::scheduleRun(long step, real gmx_unused time)
{
    bool needComReduction = doStopCM_ && do_per_step(step, nstcomm_);

    needGlobalReduction_ = needGlobalReduction_ || needComReduction || do_per_step(step, nstglobalcomm_);

    /* With Leap-Frog we can skip compute_globals at
     * non-communication steps, but we need to calculate
     * the kinetic energy one step before communication.
     */
    /* for vv, the first half of the integration actually corresponds to the previous step.
       So we need information from the last step in the first half of the integration */

    if (needGlobalReduction_ ||
        (isLF_ && do_per_step(step+1, nstglobalcomm_)) ||
        (isVV_ && do_per_step(step-1, nstglobalcomm_)))
    {
        auto returnValue = std::make_unique<ElementFunctionType>(
                std::bind(&ComputeGlobalsElement::run, this,
                        needGlobalReduction_, needEnergyReduction_, needComReduction));
        needGlobalReduction_ = false;
        needEnergyReduction_ = false;
        return returnValue;
    }

    return nullptr;
}

ElementFunctionTypePtr ComputeGlobalsElement::registerTeardown()
{
    return nullptr;
}

void ComputeGlobalsElement::globalReductionNeeded()
{
    needGlobalReduction_ = true;
}

void ComputeGlobalsElement::setup()
{
    SimulationSignaller nullSignaller(nullptr, nullptr, nullptr, false, false);
    int                 cglo_flags = (CGLO_INITIALIZATION | CGLO_TEMPERATURE | CGLO_GSTAT
                                      | (isVV_ ? CGLO_PRESSURE : 0)
                                      | (isVV_ ? CGLO_CONSTRAINT : 0)
                                      | (false ? CGLO_READEKIN : 0)); // TODO: We're not reading anything right now...

    bool bSumEkinhOld = false;

    auto x      = as_rvec_array(microState_->writePreviousPosition().paddedArrayRef().data());
    auto v      = as_rvec_array(microState_->writeVelocity().paddedArrayRef().data());
    auto box    = microState_->getBox();
    real lambda = 0;

    /* To minimize communication, compute_globals computes the COM velocity
     * and the kinetic energy for the velocities without COM motion removed.
     * Thus to get the kinetic energy without the COM contribution, we need
     * to call compute_globals twice.
     */
    for (int cgloIteration = 0; cgloIteration < (doStopCM_ ? 2 : 1); cgloIteration++)
    {
        int cglo_flags_iteration = cglo_flags;
        if (doStopCM_ && cgloIteration == 0)
        {
            cglo_flags_iteration |= CGLO_STOPCM;
            cglo_flags_iteration &= ~CGLO_TEMPERATURE;
        }
        compute_globals(fplog_, gstat_, cr_, inputrec_, fr_, ekind_,
                        x, v, box, lambda,
                        mdatoms_, nrnb_, vcm_,
                        nullptr, enerd_, force_vir_, shake_vir_, total_vir_, pres_, mu_tot_,
                        constr_, &nullSignaller, box,
                        &accumulateGlobals_,
                        &totalNumberOfBondedInteractions_, &bSumEkinhOld, cglo_flags_iteration
                        | (shouldCheckNumberOfBondedInteractions_ ? CGLO_CHECK_NUMBER_OF_BONDED_INTERACTIONS : 0));
    }
    checkNumberOfBondedInteractions(mdlog_, cr_, totalNumberOfBondedInteractions_,
                                    top_global_, top_, x, box,
                                    &shouldCheckNumberOfBondedInteractions_);
}

void ComputeGlobalsElement::run(bool needGlobalReduction, bool needEnergyReduction, bool needComReduction)
{
    // Since we're already communicating at this step, we
    // can propagate intra-simulation signals. Note that
    // check_nstglobalcomm has the responsibility for
    // choosing the value of nstglobalcomm that is one way
    // bGStat becomes true, so we can't get into a
    // situation where e.g. checkpointing can't be
    // signalled.
    gmx_multisim_t     *ms               = nullptr;
    bool                doInterSimSignal = false;
    bool                doIntraSimSignal = true;
    SimulationSignaller signaller(&signals_, cr_, ms, doInterSimSignal, doIntraSimSignal);

    bool                bSumEkinhOld = false; // Needed only for VV-AVEK, which we don't support for now

    auto                x      = as_rvec_array(microState_->writePreviousPosition().paddedArrayRef().data());
    auto                v      = as_rvec_array(microState_->writeVelocity().paddedArrayRef().data());
    auto                box    = microState_->getBox();
    real                lambda = 0;

    int                 flags =
        CGLO_TEMPERATURE |     // TODO: This used not to be done every time when doing vv - why?
        CGLO_PRESSURE |        // TODO: This used not to be done every time when doing vv - why?
        CGLO_CONSTRAINT;       // TODO: This used to be done in a second reduction with vv - check!

    if (needGlobalReduction)
    {
        flags |= CGLO_GSTAT;
    }

    if (needEnergyReduction)
    {
        flags |= CGLO_ENERGY;
    }

    if (needComReduction)
    {
        flags |= CGLO_STOPCM;
    }

    compute_globals(fplog_, gstat_, cr_, inputrec_, fr_, ekind_,
                    x, v, box, lambda,
                    mdatoms_, nrnb_, vcm_,
                    wcycle_, enerd_, force_vir_, shake_vir_, total_vir_, pres_, mu_tot_,
                    constr_, &signaller,
                    box,  // TODO: was lastbox - might be a problem when box changes!
                    &accumulateGlobals_,
                    &totalNumberOfBondedInteractions_, &bSumEkinhOld,
                    flags |
                    (shouldCheckNumberOfBondedInteractions_ ? CGLO_CHECK_NUMBER_OF_BONDED_INTERACTIONS : 0)
                    );
    checkNumberOfBondedInteractions(mdlog_, cr_, totalNumberOfBondedInteractions_,
                                    top_global_, top_, x, box,
                                    &shouldCheckNumberOfBondedInteractions_);
}

void ComputeGlobalsElement::needToCheckNumberOfBondedInteractions()
{
    shouldCheckNumberOfBondedInteractions_ = true;
}

CheckNOfBondedInteractionsCallbackPtr ComputeGlobalsElement::getCheckNOfBondedInteractionsCallback()
{
    return std::make_unique<CheckNOfBondedInteractionsCallback>(
            std::bind(&ComputeGlobalsElement::needToCheckNumberOfBondedInteractions, this));
}

EnergySignallerCallbackPtr ComputeGlobalsElement::getCalculateEnergyCallback()
{
    return std::make_unique<EnergySignallerCallback>(
            EnergySignallerCallback([this](){
                                        this->needEnergyReduction_ = true;
                                        this->needGlobalReduction_ = true;
                                    }));
}

EnergySignallerCallbackPtr ComputeGlobalsElement::getCalculateVirialCallback()
{
    return std::make_unique<EnergySignallerCallback>(
            EnergySignallerCallback([this](){this->needGlobalReduction_ = true; }));
}

EnergySignallerCallbackPtr ComputeGlobalsElement::getWriteEnergyCallback()
{
    return std::make_unique<EnergySignallerCallback>(
            EnergySignallerCallback([this](){
                                        this->needEnergyReduction_ = true;
                                        this->needGlobalReduction_ = true;
                                    }));
}

EnergySignallerCallbackPtr ComputeGlobalsElement::getCalculateFreeEnergyCallback()
{
    return nullptr;
}

}
