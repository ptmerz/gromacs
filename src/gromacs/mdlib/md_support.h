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
#ifndef GMX_MDLIB_MD_SUPPORT_H
#define GMX_MDLIB_MD_SUPPORT_H

#include "gromacs/mdlib/simulationsignal.h"
#include "gromacs/mdlib/vcm.h"
#include "gromacs/mdrun/integratorinterfaces.h"
#include "gromacs/mdrunutility/accumulateglobals.h"
#include "gromacs/timing/wallcycle.h"

struct gmx_ekindata_t;
struct gmx_enerdata_t;
struct gmx_global_stat;
struct gmx_localtop_t;
struct gmx_mtop_t;
struct gmx_multisim_t;
struct gmx_signalling_t;
struct t_extmass;
struct t_forcerec;
struct t_grpopts;
struct t_lambda;
struct t_nrnb;
class t_state;
struct t_trxframe;

namespace gmx
{
class Constraints;
class MDLogger;
class SimulationSignaller;
}

/* Define a number of flags to better control the information
 * passed to compute_globals in md.c and global_stat.
 */

/* we are initializing and not yet in the actual MD loop */
#define CGLO_INITIALIZATION (1<<1)
/* we are computing the kinetic energy from average velocities */
#define CGLO_EKINAVEVEL     (1<<2)
/* we are removing the center of mass momenta */
#define CGLO_STOPCM         (1<<3)
/* bGStat is defined in do_md */
#define CGLO_GSTAT          (1<<4)
/* Sum the energy terms in global computation */
#define CGLO_ENERGY         (1<<6)
/* Sum the kinetic energy terms in global computation */
#define CGLO_TEMPERATURE    (1<<7)
/* Sum the kinetic energy terms in global computation */
#define CGLO_PRESSURE       (1<<8)
/* Sum the constraint term in global computation */
#define CGLO_CONSTRAINT     (1<<9)
/* Reading ekin from the trajectory */
#define CGLO_READEKIN       (1<<10)
/* we need to reset the ekin rescaling factor here */
#define CGLO_SCALEEKIN      (1<<11)
/* After a new DD partitioning, we need to set a flag to schedule
 * global reduction of the total number of bonded interactions that
 * will be computed, to check none are missing. */
#define CGLO_CHECK_NUMBER_OF_BONDED_INTERACTIONS (1<<12)


/*! \brief Return the number of steps that will take place between
 * intra-simulation communications, given the constraints of the
 * inputrec and the value of mdrun -gcom. */
int check_nstglobalcomm(const gmx::MDLogger &mdlog,
                        int                  nstglobalcomm,
                        t_inputrec          *ir,
                        const t_commrec    * cr);

/*! \brief Return true if the \p value is equal across the set of multi-simulations
 *
 * \todo This duplicates some of check_multi_int. Consolidate. */
bool multisim_int_all_are_equal(const gmx_multisim_t *ms,
                                int64_t               value);

void rerun_parallel_comm(t_commrec *cr, t_trxframe *fr,
                         gmx_bool *bLastStep);

/* Set the lambda values in the global state from a frame read with rerun */
void setCurrentLambdasRerun(int64_t step, const t_lambda *fepvals,
                            const t_trxframe *rerun_fr, const double *lam0,
                            t_state *globalState);

/* Set the lambda values at each step of mdrun when they change */
void setCurrentLambdasLocal(int64_t step, const t_lambda *fepvals,
                            const double *lam0, t_state *state);

int multisim_min(const gmx_multisim_t *ms, int nmin, int n);
/* Set an appropriate value for n across the whole multi-simulation */

void compute_globals(FILE *fplog, gmx_global_stat *gstat, t_commrec *cr, t_inputrec *ir,
                     t_forcerec *fr, gmx_ekindata_t *ekind,
                     t_state *state, t_mdatoms *mdatoms,
                     t_nrnb *nrnb, t_vcm *vcm, gmx_wallcycle_t wcycle,
                     gmx_enerdata_t *enerd, tensor force_vir, tensor shake_vir, tensor total_vir,
                     tensor pres, rvec mu_tot, gmx::Constraints *constr,
                     gmx::SimulationSignaller *signalCoordinator,
                     matrix box,
                     gmx::AccumulateGlobals *accumulateGlobals,
                     int *totalNumberOfBondedInteractions,
                     gmx_bool *bSumEkinhOld, int flags);
/* Compute global variables during integration */

namespace gmx
{
class ComputeGlobalsElement : public IIntegratorElement, public IEnergySignallerClient
{
    public:
        ComputeGlobalsElement(
            StepAccessorPtr       stepAccessor,
            t_state              *localState,
            gmx_enerdata_t       *enerd,
            tensor                force_vir,
            tensor                shake_vir,
            tensor                total_vir,
            tensor                pres,
            rvec                  mu_tot,
            FILE                 *fplog,
            const MDLogger       &mdlog,
            t_commrec            *cr,
            t_inputrec           *inputrec,
            t_mdatoms            *mdatoms,
            t_nrnb               *nrnb,
            t_forcerec           *fr,
            gmx_wallcycle_t       wcycle,
            gmx_mtop_t           *global_top,
            gmx_localtop_t       *top,
            gmx_ekindata_t       *ekind,
            Constraints          *constr,
            t_vcm                *vcm,
            int                   globalCommunicationInterval);
        ~ComputeGlobalsElement() override;

        ElementFunctionTypePtr registerSetup() override;
        ElementFunctionTypePtr registerRun() override;
        ElementFunctionTypePtr registerTeardown() override;

        void globalReductionNeeded();

        CheckNOfBondedInteractionsCallbackPtr getCheckNOfBondedInteractionsCallback();

        EnergySignallerCallbackPtr getCalculateEnergyCallback() override;
        EnergySignallerCallbackPtr getCalculateVirialCallback() override;
        EnergySignallerCallbackPtr getWriteEnergyCallback() override;
        EnergySignallerCallbackPtr getCalculateFreeEnergyCallback() override;

    private:
        void setup();
        void run();
        void needToCheckNumberOfBondedInteractions();

        const bool doStopCM_;
        int        nstcomm_;
        bool       needGlobalReduction_;
        bool       needEnergyReduction_;
        int        nstglobalcomm_;
        bool       isVV_;
        bool       isLF_;

        /* Domain decomposition could incorrectly miss a bonded
           interaction, but checking for that requires a global
           communication stage, which does not otherwise happen in DD
           code. So we do that alongside the first global energy reduction
           after a new DD is made. These variables handle whether the
           check happens, and the result it returns. */
        int                    totalNumberOfBondedInteractions_;
        bool                   shouldCheckNumberOfBondedInteractions_;

        gmx_global_stat       *gstat_;
        AccumulateGlobals      accumulateGlobals_;

        StepAccessorPtr        stepAccessor_;

        // TODO: Rethink access to these
        //! Handles logging.
        FILE             *fplog_;
        //! Handles logging.
        const MDLogger   &mdlog_;
        //! Handles communication.
        t_commrec        *cr_;
        //! Contains user input mdp options.
        t_inputrec       *inputrec_;
        //! Full system topology.
        const gmx_mtop_t *top_global_;
        //! Atom parameters for this domain.
        t_mdatoms        *mdatoms_;
        //! The local state
        t_state          *localState_;
        //! Energy data structure
        gmx_enerdata_t   *enerd_;
        //! Virials
        rvec             *force_vir_, *shake_vir_, *total_vir_, *pres_;
        //! Total dipole moment (I guess...)
        real            * mu_tot_;
        //! The kinetic energy data structure
        gmx_ekindata_t   *ekind_;
        //! Handles constraints.
        Constraints      *constr_;
        //! Manages flop accounting.
        t_nrnb           *nrnb_;
        //! Manages wall cycle accounting.
        gmx_wallcycle    *wcycle_;
        //! Parameters for force calculations.
        t_forcerec       *fr_;
        //! Center of mass motion removal
        t_vcm            *vcm_;
        //! Local topology
        gmx_localtop_t   *top_;
        //! Signals
        SimulationSignals signals_;
};
}

#endif
