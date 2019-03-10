/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2018, by the GROMACS development team, led by
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
 * \brief Defines the dispatch function for the .mdp integrator field.
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_mdrun
 */
#include "gmxpre.h"

#include "integrator.h"

#include "gromacs/mdlib/stophandler.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/utility/exceptions.h"

namespace gmx
{

SimpleStepManager::SimpleStepManager(
        real timestep, long nsteps, long step, real time) :
    step_(step),
    nsteps_(nsteps + step),
    initialTime_(time),
    time_(time),
    timestep_(timestep)
{}

ElementFunctionTypePtr SimpleStepManager::registerSetup()
{
    return std::make_unique<ElementFunctionType>(
            std::bind(&SimpleStepManager::setup, this));
}
ElementFunctionTypePtr SimpleStepManager::registerRun()
{
    return std::make_unique<ElementFunctionType>(
            std::bind(&SimpleStepManager::increment, this));
}
ElementFunctionTypePtr SimpleStepManager::registerTeardown()
{
    return nullptr;
}

//! Returns a function pointer allowing to access the current step number
StepAccessorPtr SimpleStepManager::getStepAccessor() const
{
    return std::make_unique<StepAccessor>(
            [this](){return this->step_; });
}
//! Returns a function pointer allowing to access the current time
TimeAccessorPtr SimpleStepManager::getTimeAccessor() const
{
    return std::make_unique<TimeAccessor>(
            [this](){return this->time_; });
}

//! Allows clients to register a last-step callback
void SimpleStepManager::registerLastStepCallback(gmx::LastStepCallbackPtr callback)
{
    if (callback)
    {
        lastStepCallbacks_.emplace_back(std::move(callback));
    }
}

//! Increments the step and time, needs to be called every step
void SimpleStepManager::increment()
{
    ++step_;
    time_ = initialTime_ + step_*timestep_;
    if (step_ == nsteps_)
    {
        for (auto &callback : lastStepCallbacks_)
        {
            (*callback)();
        }
    }
}
//! Called before the first step - notifies clients if first step is also last step
void SimpleStepManager::setup()
{
    if (step_ == nsteps_)
    {
        for (auto &callback : lastStepCallbacks_)
        {
            (*callback)();
        }
    }
}

namespace legacy
{

//! \brief Run the correct integrator function.
void Integrator::run()
{
    switch (ei)
    {
        case eiMD:
        case eiBD:
        case eiSD1:
        case eiVV:
        case eiVVAK:
            if (!EI_DYNAMICS(ei))
            {
                GMX_THROW(APIError("do_md integrator would be called for a non-dynamical integrator"));
            }
            if (doRerun)
            {
                do_rerun();
            }
            else if (inputrec->userint1 == 99887766)
            {
                do_simple_md();
            }
            else
            {
                do_md();
            }
            break;
        case eiMimic:
            if (doRerun)
            {
                do_rerun();
            }
            else
            {
                do_mimic();
            }
            break;
        case eiSteep:
            do_steep();
            break;
        case eiCG:
            do_cg();
            break;
        case eiNM:
            do_nm();
            break;
        case eiLBFGS:
            do_lbfgs();
            break;
        case eiTPI:
        case eiTPIC:
            if (!EI_TPI(ei))
            {
                GMX_THROW(APIError("do_tpi integrator would be called for a non-TPI integrator"));
            }
            do_tpi();
            break;
        case eiSD2_REMOVED:
            GMX_THROW(NotImplementedError("SD2 integrator has been removed"));
        default:
            GMX_THROW(APIError("Non existing integrator selected"));
    }
}

Integrator::Integrator(
        FILE                               *fplog,
        t_commrec                          *cr,
        const gmx_multisim_t               *ms,
        const MDLogger                     &mdlog,
        int                                 nfile,
        const t_filenm                     *fnm,
        const gmx_output_env_t             *oenv,
        const MdrunOptions                 &mdrunOptions,
        gmx_vsite_t                        *vsite,
        Constraints                        *constr,
        gmx_enfrot                         *enforcedRotation,
        BoxDeformation                     *deform,
        IMDOutputProvider                  *outputProvider,
        t_inputrec                         *inputrec,
        gmx_mtop_t                         *top_global,
        t_fcdata                           *fcd,
        t_state                            *state_global,
        ObservablesHistory                 *observablesHistory,
        MDAtoms                            *mdAtoms,
        t_nrnb                             *nrnb,
        gmx_wallcycle                      *wcycle,
        t_forcerec                         *fr,
        PpForceWorkload                    *ppForceWorkload,
        const ReplicaExchangeParameters    &replExParams,
        gmx_membed_t                       *membed,
        AccumulateGlobalsBuilder           *accumulateGlobalsBuilder,
        gmx_walltime_accounting            *walltime_accounting,
        std::unique_ptr<StopHandlerBuilder> stopHandlerBuilder,
        int                                 ei,
        bool                                doRerun) :
    fplog(fplog),
    cr(cr),
    ms(ms),
    mdlog(mdlog),
    nfile(nfile),
    fnm(fnm),
    oenv(oenv),
    mdrunOptions(mdrunOptions),
    vsite(vsite),
    constr(constr),
    enforcedRotation(enforcedRotation),
    deform(deform),
    outputProvider(outputProvider),
    inputrec(inputrec),
    top_global(top_global),
    fcd(fcd),
    state_global(state_global),
    observablesHistory(observablesHistory),
    mdAtoms(mdAtoms),
    nrnb(nrnb),
    wcycle(wcycle),
    fr(fr),
    ppForceWorkload(ppForceWorkload),
    replExParams(replExParams),
    membed(membed),
    accumulateGlobalsBuilder_(accumulateGlobalsBuilder),
    walltime_accounting(walltime_accounting),
    stopHandlerBuilder(std::move(stopHandlerBuilder)),
    ei(ei),
    doRerun(doRerun)
{}

}  // namespace legacy

}  // namespace gmx
