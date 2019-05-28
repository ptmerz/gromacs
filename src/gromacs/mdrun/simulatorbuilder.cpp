/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2019, by the GROMACS development team, led by
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
/*! \internal
 * \brief Defines the simulator builder for mdrun
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_mdrun
 */
#include "gmxpre.h"

#include "simulatorbuilder.h"

#include "gromacs/commandline/filenm.h"
#include "gromacs/mdlib/stophandler.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdrunutility/handlerestart.h"
#include "gromacs/mdtypes/fcdata.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/observableshistory.h"

#include "legacysimulator.h"
#include "replicaexchange.h"
#include "simplesimulator.h"

namespace gmx
{
//! Build a Simulator object
std::unique_ptr<Simulator> SimulatorBuilder::build(
        FILE                               *fplog,
        t_commrec                          *cr,
        const gmx_multisim_t               *ms,
        const MDLogger                     &mdlog,
        int                                 nfile,
        const t_filenm                     *fnm,
        const gmx_output_env_t             *oenv,
        const MdrunOptions                 &mdrunOptions,
        StartingBehavior                    startingBehavior,
        gmx_vsite_t                        *vsite,
        Constraints                        *constr,
        gmx_enfrot                         *enforcedRotation,
        BoxDeformation                     *deform,
        IMDOutputProvider                  *outputProvider,
        t_inputrec                         *inputrec,
        ImdSession                         *imdSession,
        pull_t                             *pull_work,
        t_swap                             *swap,
        gmx_mtop_t                         *top_global,
        t_fcdata                           *fcd,
        t_state                            *state_global,
        ObservablesHistory                 *observablesHistory,
        MDAtoms                            *mdAtoms,
        t_nrnb                             *nrnb,
        gmx_wallcycle                      *wcycle,
        t_forcerec                         *fr,
        gmx_enerdata_t                     *enerd,
        PpForceWorkload                    *ppForceWorkload,
        const ReplicaExchangeParameters    &replExParams,
        gmx_membed_t                       *membed,
        gmx_walltime_accounting            *walltime_accounting,
        std::unique_ptr<StopHandlerBuilder> stopHandlerBuilder,
        bool                                doRerun)
{
    const auto useModularSimulator = (getenv("GMX_USE_MODULAR_SIMULATOR") != nullptr);
    if (!useModularSimulator)
    {
        return std::unique_ptr<LegacySimulator>(
                new LegacySimulator(
                        fplog, cr, ms, mdlog, nfile, fnm,
                        oenv,
                        mdrunOptions,
                        startingBehavior,
                        vsite, constr,
                        enforcedRotation,
                        deform,
                        outputProvider,
                        inputrec, imdSession, pull_work, swap, top_global,
                        fcd,
                        state_global,
                        observablesHistory,
                        mdAtoms, nrnb, wcycle, fr,
                        enerd,
                        ppForceWorkload,
                        replExParams,
                        membed,
                        walltime_accounting,
                        std::move(stopHandlerBuilder),
                        doRerun));
    }
    else
    {
        GMX_RELEASE_ASSERT(
                inputrec->eI == eiMD || inputrec->eI == eiVV,
                "Only integrators md and md-vv are supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                !doRerun,
                "Rerun is not supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                startingBehavior == StartingBehavior::NewSimulation,
                "Checkpointing is not supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                inputrec->etc == etcNO,
                "Temperature coupling is not supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                inputrec->epc == epcNO,
                "Pressure coupling is not supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                !(inputrecNptTrotter(inputrec) || inputrecNphTrotter(inputrec) || inputrecNvtTrotter(inputrec)),
                "Legacy Trotter decomposition is not supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                inputrec->efep == efepNO,
                "Free energy calculation is not supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                vsite == nullptr,
                "Virtual sites are not supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                !inputrec->bDoAwh,
                "AWH is not supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                ms == nullptr,
                "Multi-sim are not supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                replExParams.exchangeInterval == 0,
                "Replica exchange is not supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                fcd->disres.nsystems <= 1,
                "Ensemble restraints are not supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                !doSimulatedAnnealing(inputrec),
                "Simulated annealing is not supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                !inputrec->bSimTemp,
                "Simulated tempering is not supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                !inputrec->bExpanded,
                "Expanded ensemble simulations are not supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                !(opt2bSet("-ei", nfile, fnm) || observablesHistory->edsamHistory != nullptr),
                "Essential dynamics is not supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                inputrec->eSwapCoords == eswapNO,
                "Ion / water position swapping is not supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                !inputrec->bIMD,
                "Interactive MD is not supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                membed == nullptr,
                "Membrane embedding is not supported by the modular simulator.");
        GMX_RELEASE_ASSERT(
                getenv("GMX_INTEGRATE_GPU") == nullptr,
                "Integration on the GPU is not supported by the modular simulator.");

        return std::unique_ptr<SimpleSimulator>(
                new SimpleSimulator(
                        fplog, cr, ms, mdlog, nfile, fnm,
                        oenv,
                        mdrunOptions,
                        startingBehavior,
                        vsite, constr,
                        enforcedRotation,
                        deform,
                        outputProvider,
                        inputrec, imdSession, pull_work, swap, top_global,
                        fcd,
                        state_global,
                        observablesHistory,
                        mdAtoms, nrnb, wcycle, fr,
                        enerd,
                        ppForceWorkload,
                        replExParams,
                        membed,
                        walltime_accounting,
                        std::move(stopHandlerBuilder),
                        doRerun));
    }
}
}
