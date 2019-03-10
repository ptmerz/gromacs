/*
 * This file is part of the GROMACS molecular simulation package.
 *
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

#include "trajectory_writing.h"

#include "gromacs/commandline/filenm.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/fileio/tngio.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/energyoutput.h"
#include "gromacs/mdlib/mdoutf.h"
#include "gromacs/mdlib/mdrun.h"
#include "gromacs/mdlib/sim_util.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/observableshistory.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/smalloc.h"

void
do_md_trajectory_writing(FILE                     *fplog,
                         t_commrec                *cr,
                         int                       nfile,
                         const t_filenm            fnm[],
                         int64_t                   step,
                         int64_t                   step_rel,
                         double                    t,
                         t_inputrec               *ir,
                         t_state                  *state,
                         t_state                  *state_global,
                         ObservablesHistory       *observablesHistory,
                         gmx_mtop_t               *top_global,
                         t_forcerec               *fr,
                         gmx_mdoutf_t              outf,
                         const gmx::EnergyOutput  &energyOutput,
                         gmx_ekindata_t           *ekind,
                         gmx::ArrayRef<gmx::RVec>  f,
                         gmx_bool                  bCPT,
                         gmx_bool                  bRerunMD,
                         gmx_bool                  bLastStep,
                         gmx_bool                  bDoConfOut,
                         gmx_bool                  bSumEkinhOld
                         )
{
    int   mdof_flags;
    rvec *x_for_confout = nullptr;

    mdof_flags = 0;
    if (do_per_step(step, ir->nstxout))
    {
        mdof_flags |= MDOF_X;
    }
    if (do_per_step(step, ir->nstvout))
    {
        mdof_flags |= MDOF_V;
    }
    if (do_per_step(step, ir->nstfout))
    {
        mdof_flags |= MDOF_F;
    }
    if (do_per_step(step, ir->nstxout_compressed))
    {
        mdof_flags |= MDOF_X_COMPRESSED;
    }
    if (bCPT)
    {
        mdof_flags |= MDOF_CPT;
    }
    if (do_per_step(step, mdoutf_get_tng_box_output_interval(outf)))
    {
        mdof_flags |= MDOF_BOX;
    }
    if (do_per_step(step, mdoutf_get_tng_lambda_output_interval(outf)))
    {
        mdof_flags |= MDOF_LAMBDA;
    }
    if (do_per_step(step, mdoutf_get_tng_compressed_box_output_interval(outf)))
    {
        mdof_flags |= MDOF_BOX_COMPRESSED;
    }
    if (do_per_step(step, mdoutf_get_tng_compressed_lambda_output_interval(outf)))
    {
        mdof_flags |= MDOF_LAMBDA_COMPRESSED;
    }

#if GMX_FAHCORE
    if (bLastStep)
    {
        /* Enforce writing positions and velocities at end of run */
        mdof_flags |= (MDOF_X | MDOF_V);
    }
    if (MASTER(cr))
    {
        fcReportProgress( ir->nsteps, step );
    }

#if defined(__native_client__)
    fcCheckin(MASTER(cr));
#endif

    /* sync bCPT and fc record-keeping */
    if (bCPT && MASTER(cr))
    {
        fcRequestCheckPoint();
    }
#endif

    if (mdof_flags != 0)
    {
        wallcycle_start(mdoutf_get_wcycle(outf), ewcTRAJ);
        if (bCPT)
        {
            if (MASTER(cr))
            {
                if (bSumEkinhOld)
                {
                    state_global->ekinstate.bUpToDate = FALSE;
                }
                else
                {
                    update_ekinstate(&state_global->ekinstate, ekind);
                    state_global->ekinstate.bUpToDate = TRUE;
                }

                energyOutput.fillEnergyHistory(observablesHistory->energyHistory.get());
            }
        }
        mdoutf_write_to_trajectory_files(fplog, cr, outf, mdof_flags, top_global->natoms,
                                         step, t, state, state_global, observablesHistory, f);
        if (bLastStep && step_rel == ir->nsteps &&
            bDoConfOut && MASTER(cr) &&
            !bRerunMD)
        {
            if (fr->bMolPBC && state == state_global)
            {
                /* This (single-rank) run needs to allocate a
                   temporary array of size natoms so that any
                   periodicity removal for mdrun -confout does not
                   perturb the update and thus the final .edr
                   output. This makes .cpt restarts look binary
                   identical, and makes .edr restarts binary
                   identical. */
                snew(x_for_confout, state_global->natoms);
                copy_rvecn(state_global->x.rvec_array(), x_for_confout, 0, state_global->natoms);
            }
            else
            {
                /* With DD, or no bMolPBC, it doesn't matter if
                   we change state_global->x.rvec_array() */
                x_for_confout = state_global->x.rvec_array();
            }

            /* x and v have been collected in mdoutf_write_to_trajectory_files,
             * because a checkpoint file will always be written
             * at the last step.
             */
            fprintf(stderr, "\nWriting final coordinates.\n");
            if (fr->bMolPBC && !ir->bPeriodicMols)
            {
                /* Make molecules whole only for confout writing */
                do_pbc_mtop(fplog, ir->ePBC, state->box, top_global, x_for_confout);
            }
            write_sto_conf_mtop(ftp2fn(efSTO, nfile, fnm),
                                *top_global->name, top_global,
                                x_for_confout, state_global->v.rvec_array(),
                                ir->ePBC, state->box);
            if (fr->bMolPBC && state == state_global)
            {
                sfree(x_for_confout);
            }
        }
        wallcycle_stop(mdoutf_get_wcycle(outf), ewcTRAJ);
    }
}

namespace gmx
{
TrajectoryWriter::TrajectoryWriter(
        FILE *fplog, int nfile, const t_filenm fnm[],
        const MdrunOptions &mdrunOptions,
        const t_commrec *cr,
        gmx::IMDOutputProvider *outputProvider,
        const t_inputrec *ir, gmx_mtop_t *top_global,
        const gmx_output_env_t *oenv, gmx_wallcycle_t wcycle)
{
    outf_ = init_mdoutf(
                fplog, nfile, fnm, mdrunOptions, cr,
                outputProvider, ir, top_global, oenv, wcycle);
}

void TrajectoryWriter::setup()
{
    for (auto &callback : setupCallbacks_)
    {
        (*callback)(outf_);
    }
}

void TrajectoryWriter::write()
{
    for (auto &callback : runTrajectoryCallbacks_)
    {
        (*callback)(outf_);
    }
    for (auto &callback : runEnergyCallbacks_)
    {
        (*callback)(outf_);
    }
}

void TrajectoryWriter::teardown()
{
    for (auto &callback : teardownCallbacks_)
    {
        (*callback)(outf_);
    }
    mdoutf_tng_close(outf_);
    done_mdoutf(outf_);
}

ElementFunctionTypePtr TrajectoryWriter::registerSetup()
{
    return std::make_unique<ElementFunctionType>(
            std::bind(&TrajectoryWriter::setup, this));
}

ElementFunctionTypePtr TrajectoryWriter::registerRun()
{
    return std::make_unique<ElementFunctionType>(
            std::bind(&TrajectoryWriter::write, this));
}


ElementFunctionTypePtr TrajectoryWriter::registerTeardown()
{
    return std::make_unique<ElementFunctionType>(
            std::bind(&TrajectoryWriter::teardown, this));
}

void TrajectoryWriter::registerClient(
        TrajectoryWriterCallbackPtr setupCallback,
        TrajectoryWriterCallbackPtr runTrajectoryCallback,
        TrajectoryWriterCallbackPtr runEnergyCallback,
        TrajectoryWriterCallbackPtr teardownCallback)
{
    if (setupCallback)
    {
        setupCallbacks_.emplace_back(std::move(setupCallback));
    }
    if (runTrajectoryCallback)
    {
        runTrajectoryCallbacks_.emplace_back(std::move(runTrajectoryCallback));
    }
    if (runEnergyCallback)
    {
        runEnergyCallbacks_.emplace_back(std::move(runEnergyCallback));
    }
    if (teardownCallback)
    {
        teardownCallbacks_.emplace_back(std::move(teardownCallback));
    }
}

TrajectorySignaller::TrajectorySignaller(
        StepAccessorPtr stepAccessor,
        int nstxout, int nstvout, int nstfout, int nstxout_compressed) :
    stepAccessor_(std::move(stepAccessor)),
    nstxout_(nstxout),
    nstvout_(nstvout),
    nstfout_(nstfout),
    nstxout_compressed_(nstxout_compressed)
{}

ElementFunctionTypePtr TrajectorySignaller::registerSetup()
{
    return nullptr;
}

ElementFunctionTypePtr TrajectorySignaller::registerRun()
{
    return std::make_unique<ElementFunctionType>(
            std::bind(&TrajectorySignaller::run, this));
}

ElementFunctionTypePtr TrajectorySignaller::registerTeardown()
{
    return nullptr;
}

void TrajectorySignaller::registerCallback(TrajectorySignallerCallbackPtr callback)
{
    if (callback)
    {
        callbacks_.emplace_back(std::move(callback));
    }
}

void TrajectorySignaller::run()
{
    auto currentStep = (*stepAccessor_)();
    if (do_per_step(currentStep, nstxout_) ||
        do_per_step(currentStep, nstvout_) ||
        do_per_step(currentStep, nstfout_) ||
        do_per_step(currentStep, nstxout_compressed_))
    {
        for (const auto &callback : callbacks_)
        {
            (*callback)();
        }
    }
}
}
