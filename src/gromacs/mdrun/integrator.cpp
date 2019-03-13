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

#include "gromacs/domdec/domdec.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/constr.h"
#include "gromacs/mdlib/energyoutput.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdlib/md_support.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdlib/mdrun.h"
#include "gromacs/mdlib/shellfc.h"
#include "gromacs/mdlib/sim_util.h"
#include "gromacs/mdlib/stophandler.h"
#include "gromacs/mdlib/trajectory_writing.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdlib/tgroup.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/logger.h"

namespace gmx
{

SimpleIntegrator::SimpleIntegrator(
        FILE                    *fplog,
        t_commrec               *cr,
        const MdrunOptions      &mdrunOptions,
        BoxDeformation          *deform,
        t_inputrec              *inputrec,
        gmx_mtop_t              *top_global,
        t_nrnb                  *nrnb,
        Constraints             *constr,
        gmx_wallcycle           *wcycle,
        gmx_walltime_accounting *walltime_accounting,
        StepAccessorPtr          stepAccessor) :
    initStep_(inputrec->init_step),
    stepAccessor_(std::move(stepAccessor)),
    fplog_(fplog),
    cr_(cr),
    wcycle_(wcycle),
    walltime_accounting_(walltime_accounting),
    localTopology_(new gmx_localtop_t()),
    shellfc_(nullptr),
    do_verbose_(false),
    upd_(new Update(inputrec, deform)),
    eKinData_(std::make_unique<gmx_ekindata_t>()),
    vcm_(new t_vcm(top_global->groups, *inputrec))
{
    if (!mdrunOptions.continuationOptions.appendFiles)
    {
        pleaseCiteCouplingAlgorithms(fplog, *inputrec);
    }

    /* Energy terms and groups */
    snew(enerd_, 1);
    init_enerdata(top_global->groups.grps[egcENER].nr, inputrec->fepvals->n_lambda,
                  enerd_);

    init_ekindata(fplog, top_global, &(inputrec->opts), eKinData_.get());
    /* Copy the cos acceleration to the groups struct */
    eKinData_->cosacc.cos_accel = inputrec->cos_accel;

    init_nrnb(nrnb);

    clear_mat(force_vir_);
    clear_mat(shake_vir_);
    clear_mat(total_vir_);
    clear_mat(pres_);

    reportComRemovalInfo(fplog, *vcm_);

    // Check for polarizable models and flexible constraints
    shellfc_ = init_shell_flexcon(fplog,
                                  top_global, constr ? constr->numFlexibleConstraints() : 0,
                                  inputrec->nstcalcenergy, DOMAINDECOMP(cr));

    if (MASTER(cr))
    {
        char tbuf[20];
        char sbuf[STEPSTRSIZE], sbuf2[STEPSTRSIZE];
        fprintf(stderr, "starting mdrun '%s'\n",
                *(top_global->name));
        if (inputrec->nsteps >= 0)
        {
            sprintf(tbuf, "%8.1f", (inputrec->init_step+inputrec->nsteps)*inputrec->delta_t);
        }
        else
        {
            sprintf(tbuf, "%s", "infinite");
        }
        if (inputrec->init_step > 0)
        {
            fprintf(stderr, "%s steps, %s ps (continuing from step %s, %8.1f ps).\n",
                    gmx_step_str(inputrec->init_step+inputrec->nsteps, sbuf), tbuf,
                    gmx_step_str(inputrec->init_step, sbuf2),
                    inputrec->init_step*inputrec->delta_t);
        }
        else
        {
            fprintf(stderr, "%s steps, %s ps.\n",
                    gmx_step_str(inputrec->nsteps, sbuf), tbuf);
        }
        fprintf(fplog, "\n");
    }
}

void SimpleIntegrator::setup(std::unique_ptr<gmx::IntegratorLoop> outerLoop,
                             std::shared_ptr<gmx::MicroState>     microState)
{
    outerLoop_  = std::move(outerLoop);
    microState_ = std::move(microState);
}

void SimpleIntegrator::run()
{
    walltime_accounting_start_time(walltime_accounting_);
    wallcycle_start(wcycle_, ewcRUN);
    print_start(fplog_, cr_, walltime_accounting_, "mdrun");

    auto setup    = outerLoop_->registerSetup();
    auto run      = outerLoop_->registerRun();
    auto teardown = outerLoop_->registerTeardown();

    // uncrustify wants these shifted by one space...
     (*setup)();
     (*run)();
     (*teardown)();

    walltime_accounting_end_time(walltime_accounting_);
    walltime_accounting_set_nsteps_done(walltime_accounting_, (*stepAccessor_)() - initStep_);
}

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

IntegratorLoop::IntegratorLoop(
        std::vector < std::unique_ptr < IIntegratorElement>> loopElements,
        long                                                 numSteps) :
    loopElements_(std::move(loopElements)),
    numSteps_(numSteps),
    isLastStep_(false)
{
    for (auto &element : loopElements_)
    {
        auto setup = element->registerSetup();
        if (setup)
        {
            setupFunctions_.emplace_back(std::move(setup));
        }

        auto run = element->registerRun();
        if (run)
        {
            runFunctions_.emplace_back(std::move(run));
        }

        auto teardown = element->registerTeardown();
        if (teardown)
        {
            teardownFunctions_.emplace_back(std::move(teardown));
        }
    }
}

void IntegratorLoop::setup()
{
    for (const auto &setupFct : this->setupFunctions_)
    {
        (*setupFct)();
    }
}
void IntegratorLoop::run()
{
    long step = 0;
    while (!isLastStep_ && (numSteps_ < 0 || step < numSteps_))
    {
        for (const auto &runFct : this->runFunctions_)
        {
            (*runFct)();
        }
        ++step;
    }
}
void IntegratorLoop::teardown()
{
    // Perform last step
    for (const auto &runFct : this->runFunctions_)
    {
        (*runFct)();
    }
    for (const auto &teardownFct : this->teardownFunctions_)
    {
        (*teardownFct)();
    }
}

ElementFunctionTypePtr IntegratorLoop::registerSetup()
{
    return std::make_unique<ElementFunctionType>(
            std::bind(&IntegratorLoop::setup, this));
}
ElementFunctionTypePtr IntegratorLoop::registerRun()
{
    return std::make_unique<ElementFunctionType>(
            std::bind(&IntegratorLoop::run, this));
}
ElementFunctionTypePtr IntegratorLoop::registerTeardown()
{
    return std::make_unique<ElementFunctionType>(
            std::bind(&IntegratorLoop::teardown, this));
}

void IntegratorLoop::nextStepIsLast()
{
    isLastStep_ = true;
}

LastStepCallbackPtr IntegratorLoop::getLastStepCallback()
{
    return std::make_unique<LastStepCallback>(
            std::bind(&IntegratorLoop::nextStepIsLast, this));
}

void IntegratorLoopBuilder::addElement(std::unique_ptr<gmx::IIntegratorElement> element)
{
    elements_.emplace_back(std::move(element));
}
std::unique_ptr<IntegratorLoop> IntegratorLoopBuilder::build(long numSteps)
{
    return std::unique_ptr<IntegratorLoop>(
            new IntegratorLoop(std::move(elements_), numSteps));
}

IntegratorBuilder::IntegratorBuilder(
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
    fplog_(fplog),
    cr_(cr),
    ms_(ms),
    mdlog_(mdlog),
    nfile_(nfile),
    fnm_(fnm),
    oenv_(oenv),
    mdrunOptions_(mdrunOptions),
    vsite_(vsite),
    constr_(constr),
    enforcedRotation_(enforcedRotation),
    deform_(deform),
    outputProvider_(outputProvider),
    inputrec_(inputrec),
    top_global_(top_global),
    fcd_(fcd),
    state_global_(state_global),
    observablesHistory_(observablesHistory),
    mdAtoms_(mdAtoms),
    nrnb_(nrnb),
    wcycle_(wcycle),
    fr_(fr),
    ppForceWorkload_(ppForceWorkload),
    replExParams_(replExParams),
    membed_(membed),
    accumulateGlobalsBuilder_(accumulateGlobalsBuilder),
    walltime_accounting_(walltime_accounting),
    stopHandlerBuilder_(std::move(stopHandlerBuilder)),
    ei_(ei),
    doRerun_(doRerun)
{}

std::unique_ptr<Integrator> IntegratorBuilder::build()
{
    // hack for legacy integrator - will be replaced by feature switch
    if (inputrec_->userint1 != 99887766)
    {
        return std::make_unique<legacy::Integrator>(
                fplog_, cr_, ms_, mdlog_, nfile_, fnm_, oenv_, mdrunOptions_, vsite_, constr_,
                enforcedRotation_, deform_, outputProvider_, inputrec_, top_global_, fcd_, state_global_,
                observablesHistory_, mdAtoms_, nrnb_, wcycle_, fr_, ppForceWorkload_, replExParams_,
                membed_, accumulateGlobalsBuilder_, walltime_accounting_, std::move(stopHandlerBuilder_),
                ei_, doRerun_);
    }

    /*
     * Build step manager
     */
    auto stepManager = std::make_unique<SimpleStepManager>(
                inputrec_->delta_t, inputrec_->nsteps, inputrec_->init_step, inputrec_->init_t);

    /*
     * Instantiate integrator
     */
    auto integrator = std::unique_ptr<SimpleIntegrator>(
                new SimpleIntegrator(
                        fplog_, cr_, mdrunOptions_, deform_, inputrec_,
                        top_global_, nrnb_, constr_,
                        wcycle_, walltime_accounting_, stepManager->getStepAccessor()));

    auto microState = std::make_shared<MicroState>(
                stepManager->getStepAccessor(), stepManager->getTimeAccessor(),
                top_global_->natoms, fplog_, cr_, state_global_,
                inputrec_->nstxout, inputrec_->nstvout, inputrec_->nstfout, inputrec_->nstxout_compressed);

    /*
     * Build neighbor search signaller
     */
    auto neighborSearchSignaller = std::make_unique<NeighborSearchSignaller>(
                stepManager->getStepAccessor(), inputrec_->nstlist);

    /*
     * Build logging signaller
     */
    auto logSignaller = std::make_unique<LoggingSignaller>(
                stepManager->getStepAccessor(), inputrec_->nstlog);
    stepManager->registerLastStepCallback(logSignaller->getLastStepCallback());

    /*
     * Build energy signaller
     */
    auto energySignaller = std::make_unique<EnergySignaller>(
                stepManager->getStepAccessor(),
                inputrec_->nstcalcenergy, inputrec_->nstenergy);
    stepManager->registerLastStepCallback(energySignaller->getLastStepCallback());

    /*
     * Build trajectory signaller
     */
    auto trajectorySignaller = std::make_unique<TrajectorySignaller>(
                stepManager->getStepAccessor(),
                inputrec_->nstxout, inputrec_->nstvout, inputrec_->nstfout, inputrec_->nstxout_compressed);

    /*
     * Build compute globals element
     */
    auto computeGlobalsElement = std::make_unique<ComputeGlobalsElement>(
                stepManager->getStepAccessor(), microState, integrator->enerd_,
                integrator->force_vir_, integrator->shake_vir_, integrator->total_vir_,
                integrator->pres_, integrator->mu_tot_, fplog_, mdlog_, cr_, inputrec_,
                mdAtoms_->mdatoms(), nrnb_, fr_, wcycle_, top_global_, integrator->localTopology_,
                integrator->eKinData_.get(), constr_, integrator->vcm_, mdrunOptions_.globalCommunicationInterval);
    energySignaller->registerCallback(
            computeGlobalsElement->getCalculateEnergyCallback(),
            computeGlobalsElement->getCalculateVirialCallback(),
            computeGlobalsElement->getWriteEnergyCallback(),
            computeGlobalsElement->getCalculateFreeEnergyCallback());

    /*
     * Build the domdec element (needs valid state pointer)
     */
    const int nstglobalcomm = check_nstglobalcomm(
                mdlog_, mdrunOptions_.globalCommunicationInterval, inputrec_, cr_);
    auto      domDecElement = std::make_unique<DomDecElement>(
                nstglobalcomm, stepManager->getStepAccessor(),
                mdrunOptions_.verbose, mdrunOptions_.verboseStepPrintInterval,
                microState, fplog_, cr_, mdlog_, constr_, inputrec_, top_global_,
                mdAtoms_, nrnb_, wcycle_, fr_, integrator->localTopology_,
                integrator->shellfc_, integrator->upd_,
                computeGlobalsElement->getCheckNOfBondedInteractionsCallback());
    neighborSearchSignaller->registerCallback(domDecElement->getNSCallback());

    /*
     * Trajectory / Energy
     */
    auto trajectoryWriter = std::make_unique<TrajectoryWriter>(
                fplog_, nfile_, fnm_, mdrunOptions_, cr_, outputProvider_,
                inputrec_, top_global_, oenv_, wcycle_);

    auto energyElement = std::make_unique<EnergyElement>(
                stepManager->getStepAccessor(), stepManager->getTimeAccessor(),
                microState, top_global_, inputrec_, mdAtoms_,
                integrator->enerd_, integrator->force_vir_, integrator->shake_vir_,
                integrator->total_vir_, integrator->pres_,
                integrator->eKinData_.get(), constr_, integrator->mu_tot_, fplog_, fcd_, MASTER(cr_));
    // Could be done by a builder
    trajectoryWriter->registerClient(
            energyElement->registerTrajectoryWriterSetup(),
            energyElement->registerTrajectoryRun(),
            energyElement->registerEnergyRun(),
            energyElement->registerTrajectoryWriterTeardown());
    stepManager->registerLastStepCallback(energyElement->getLastStepCallback());
    energySignaller->registerCallback(
            energyElement->getCalculateEnergyCallback(),
            energyElement->getCalculateVirialCallback(),
            energyElement->getWriteEnergyCallback(),
            energyElement->getCalculateFreeEnergyCallback());
    logSignaller->registerCallback(energyElement->getLoggingCallback());

    // Could be done by a builder
    trajectoryWriter->registerClient(
            microState->registerTrajectoryWriterSetup(),
            microState->registerTrajectoryRun(),
            microState->registerEnergyRun(),
            microState->registerTrajectoryWriterTeardown());
    trajectorySignaller->registerCallback(microState->getTrajectorySignallerCallback());

    std::unique_ptr<IIntegratorElement> forceElement;

    if (integrator->shellfc_)
    {
        auto fElement = std::make_unique<ShellFCElement>(
                    inputrecDynamicBox(inputrec_), DOMAINDECOMP(cr_), mdrunOptions_.verbose,
                    stepManager->getStepAccessor(), stepManager->getTimeAccessor(),
                    microState, integrator->enerd_, integrator->force_vir_, integrator->mu_tot_,
                    fplog_, cr_, inputrec_, mdAtoms_->mdatoms(), nrnb_,
                    fr_, fcd_, wcycle_, integrator->localTopology_, &top_global_->groups,
                    constr_, integrator->shellfc_, top_global_, ppForceWorkload_);

        energySignaller->registerCallback(
                fElement->getCalculateEnergyCallback(),
                fElement->getCalculateVirialCallback(),
                fElement->getWriteEnergyCallback(),
                fElement->getCalculateFreeEnergyCallback());

        neighborSearchSignaller->registerCallback(
                fElement->getNSCallback());

        forceElement = std::move(fElement);
    }
    else
    {
        auto fElement = std::make_unique<ForceElement>(
                    inputrecDynamicBox(inputrec_), DOMAINDECOMP(cr_),
                    stepManager->getStepAccessor(), stepManager->getTimeAccessor(),
                    microState, integrator->enerd_, integrator->force_vir_, integrator->mu_tot_,
                    fplog_, cr_, inputrec_, mdAtoms_->mdatoms(), nrnb_,
                    fr_, fcd_, wcycle_, integrator->localTopology_, &top_global_->groups, ppForceWorkload_);

        energySignaller->registerCallback(
                fElement->getCalculateEnergyCallback(),
                fElement->getCalculateVirialCallback(),
                fElement->getWriteEnergyCallback(),
                fElement->getCalculateFreeEnergyCallback());

        neighborSearchSignaller->registerCallback(
                fElement->getNSCallback());

        forceElement = std::move(fElement);
    }

    /*
     * Build propagators
     */
    std::unique_ptr<UpdateStep> updateStep1;
    std::unique_ptr<UpdateStep> updateStep2;

    if (inputrec_->eI == eiVV)
    {
        UpdateStepBuilder updateStepBuilder1;
        auto              updateVelocity1 = std::make_unique<UpdateVelocity>(
                    inputrec_->delta_t / 2.0, microState,
                    inputrec_->opts.acc, inputrec_->opts.nFreeze,
                    mdAtoms_->mdatoms());
        updateStepBuilder1.addUpdateElement(std::move(updateVelocity1));

        updateStep1 = updateStepBuilder1.build(mdAtoms_->mdatoms(), wcycle_);

        UpdateStepBuilder updateStepBuilder2;
        auto              updateVelocity2 = std::make_unique<UpdateVelocity>(
                    inputrec_->delta_t / 2.0, microState,
                    inputrec_->opts.acc, inputrec_->opts.nFreeze,
                    mdAtoms_->mdatoms());
        updateStepBuilder2.addUpdateElement(std::move(updateVelocity2));

        auto updatePosition = std::make_unique<UpdatePosition>(
                    inputrec_->delta_t, microState,
                    inputrec_->opts.nFreeze, mdAtoms_->mdatoms(), integrator->upd_);
        updateStepBuilder2.addUpdateElement(std::move(updatePosition));

        updateStep2 = updateStepBuilder2.build(mdAtoms_->mdatoms(), wcycle_);
    }
    else if (inputrec_->eI == eiMD)
    {
        UpdateStepBuilder updateStepBuilder1;
        updateStep1 = updateStepBuilder1.build(mdAtoms_->mdatoms(), wcycle_);

        UpdateStepBuilder updateStepBuilder2;
        auto              updateLeapFrog = std::make_unique<UpdateLeapfrog>(
                    inputrec_->delta_t, stepManager->getStepAccessor(), microState,
                    inputrec_, mdAtoms_->mdatoms(), integrator->eKinData_.get(),
                    integrator->upd_);
        updateStepBuilder2.addUpdateElement(std::move(updateLeapFrog));

        updateStep2 = updateStepBuilder2.build(mdAtoms_->mdatoms(), wcycle_);
    }
    else
    {
        gmx_fatal(FARGS, "Unknown integrator.");
    }

    auto constrainCoordinates = std::make_unique<ConstrainCoordinates>(
                stepManager->getStepAccessor(), microState,
                mdAtoms_->mdatoms(), integrator->upd_,
                integrator->shake_vir_, constr_, fplog_, inputrec_);
    logSignaller->registerCallback(constrainCoordinates->getLoggingCallback());
    energySignaller->registerCallback(
            constrainCoordinates->getCalculateEnergyCallback(),
            constrainCoordinates->getCalculateVirialCallback(),
            constrainCoordinates->getWriteEnergyCallback(),
            constrainCoordinates->getCalculateFreeEnergyCallback());

    std::unique_ptr<ConstrainVelocities> constrainVelocities;
    if (inputrec_->eI == eiVV)
    {
        constrainVelocities = std::make_unique<ConstrainVelocities>(
                    stepManager->getStepAccessor(), microState,
                    integrator->shake_vir_, constr_);
        logSignaller->registerCallback(constrainVelocities->getLoggingCallback());
        energySignaller->registerCallback(
                constrainVelocities->getCalculateEnergyCallback(),
                constrainVelocities->getCalculateVirialCallback(),
                constrainVelocities->getWriteEnergyCallback(),
                constrainVelocities->getCalculateFreeEnergyCallback());
    }

    auto finishUpdate = std::make_unique<FinishUpdateElement>(
                microState, mdAtoms_->mdatoms(), integrator->upd_,
                inputrec_, wcycle_, constr_);

    /*
     * Pre and post loop elements
     */
    auto preLoopElement  = std::make_unique<PreLoopElement>(wcycle_);
    auto postLoopElement = std::make_unique<PostLoopElement>(
                stepManager->getStepAccessor(),
                mdrunOptions_.verbose, mdrunOptions_.verboseStepPrintInterval,
                fplog_, inputrec_, cr_, walltime_accounting_, wcycle_);

    // We want to move the stepManager into the loop, but we also need to register
    // the loops to the step manager. We need to think how to solve that more elegantly -
    // for now, let's keep a function object to use the registration function.
    auto lastStepCallbackRegistration = std::bind(
                &SimpleStepManager::registerLastStepCallback, stepManager.get(), std::placeholders::_1);

    auto innerLoopBuilder = std::make_unique<IntegratorLoopBuilder>();

    innerLoopBuilder->addElement(std::move(logSignaller));
    innerLoopBuilder->addElement(std::move(energySignaller));
    innerLoopBuilder->addElement(std::move(preLoopElement));

    innerLoopBuilder->addElement(std::move(forceElement));
    innerLoopBuilder->addElement(std::move(updateStep1));
    if (inputrec_->eI == eiVV)
    {
        innerLoopBuilder->addElement(std::move(constrainVelocities));
        innerLoopBuilder->addElement(std::move(computeGlobalsElement));
        innerLoopBuilder->addElement(
                std::make_unique<ResetVForVV>(microState));
    }
    innerLoopBuilder->addElement(std::move(trajectorySignaller));
    innerLoopBuilder->addElement(std::move(updateStep2));
    innerLoopBuilder->addElement(std::move(constrainCoordinates));
    innerLoopBuilder->addElement(std::move(finishUpdate));

    if (inputrec_->eI == eiMD)
    {
        innerLoopBuilder->addElement(std::move(computeGlobalsElement));
    }

    innerLoopBuilder->addElement(std::move(energyElement));
    innerLoopBuilder->addElement(std::move(trajectoryWriter));

    innerLoopBuilder->addElement(std::move(stepManager));
    innerLoopBuilder->addElement(std::move(postLoopElement));

    auto innerLoop = innerLoopBuilder->build(inputrec_->nstlist);
    lastStepCallbackRegistration(innerLoop->getLastStepCallback());

    auto outerLoopBuilder = std::make_unique<IntegratorLoopBuilder>();
    outerLoopBuilder->addElement(std::move(neighborSearchSignaller));
    outerLoopBuilder->addElement(std::move(domDecElement));
    outerLoopBuilder->addElement(std::move(innerLoop));

    auto outerLoop = outerLoopBuilder->build();
    lastStepCallbackRegistration(outerLoop->getLastStepCallback());
    integrator->setup(std::move(outerLoop), microState);

    return std::move(integrator);
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

}   // namespace legacy

PreLoopElement::PreLoopElement(gmx_wallcycle *wcycle) :
    wcycle_(wcycle)
{}

ElementFunctionTypePtr PreLoopElement::registerSetup()
{
    return nullptr;
}

ElementFunctionTypePtr PreLoopElement::registerRun()
{
    return std::make_unique<ElementFunctionType>(
            std::bind(&PreLoopElement::run, this));
}

ElementFunctionTypePtr PreLoopElement::registerTeardown()
{
    return nullptr;
}

void PreLoopElement::run()
{
    wallcycle_start(wcycle_, ewcSTEP);
}

PostLoopElement::PostLoopElement(
        StepAccessorPtr          stepAccessor,
        bool                     doVerbose,
        int                      verboseStepInterval,
        FILE                    *fplog,
        const t_inputrec        *inputrec,
        const t_commrec         *cr,
        gmx_walltime_accounting *walltime_accounting,
        gmx_wallcycle           *wcycle) :
    isMaster_(MASTER(cr)),
    doDomainDecomposition_(DOMAINDECOMP(cr)),
    doVerbose_(doVerbose),
    verboseStepInterval_(verboseStepInterval),
    isLoggingStep_(false),
    isFirstStep_(true),
    isLastStep_(false),
    stepAccessor_(std::move(stepAccessor)),
    fplog_(fplog),
    inputrec_(inputrec),
    cr_(cr),
    walltime_accounting_(walltime_accounting),
    wcycle_(wcycle)
{}

ElementFunctionTypePtr PostLoopElement::registerSetup()
{
    return nullptr;
}

ElementFunctionTypePtr PostLoopElement::registerRun()
{
    return std::make_unique<ElementFunctionType>(
            std::bind(&PostLoopElement::run, this));
}

ElementFunctionTypePtr PostLoopElement::registerTeardown()
{
    return nullptr;
}

void PostLoopElement::run()
{
    auto cycles = wallcycle_stop(wcycle_, ewcSTEP);
    if (doDomainDecomposition_ && wcycle_)
    {
        dd_cycles_add(cr_->dd, cycles, ddCyclStep);
    }

    if (isMaster_)
    {
        auto step = (*stepAccessor_)();
        if (isLoggingStep_)
        {
            if (fflush(fplog_) != 0)
            {
                gmx_fatal(FARGS, "Cannot flush logfile - maybe you are out of disk space?");
            }
        }

        auto doVerboseThisStep = doVerbose_ &&
            (step % verboseStepInterval_ == 0 || isFirstStep_ || isLastStep_);

        /* Print the remaining wall clock time for the run */
        if (doVerboseThisStep || gmx_got_usr_signal())
        {
            print_time(stderr, walltime_accounting_, step, inputrec_, cr_);
            isFirstStep_ = false;
        }
    }
}

LastStepCallbackPtr PostLoopElement::getLastStepCallback()
{
    return std::make_unique<LastStepCallback>(
            std::bind(&PostLoopElement::nextStepIsLast, this));
}

LoggingSignallerCallbackPtr PostLoopElement::getLoggingCallback()
{
    return std::make_unique<LoggingSignallerCallback>(
            std::bind(&PostLoopElement::thisStepIsLoggingStep, this));
}

void PostLoopElement::nextStepIsLast()
{
    isLastStep_ = true;
}

void PostLoopElement::thisStepIsLoggingStep()
{
    isLoggingStep_ = true;
}
}  // namespace gmx
