/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2015,2016,2017,2018,2019, by the GROMACS development team, led by
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
 * \brief Declares the integrator interface for mdrun
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_mdrun
 */
#ifndef GMX_MDRUN_INTEGRATOR_H
#define GMX_MDRUN_INTEGRATOR_H

#include <cstdio>

#include <memory>
#include <vector>

#include "gromacs/mdrun/integratorinterfaces.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

class energyhistory_t;
struct gmx_enfrot;
struct gmx_mtop_t;
struct gmx_membed_t;
struct gmx_multisim_t;
struct gmx_output_env_t;
struct gmx_vsite_t;
struct gmx_wallcycle;
struct gmx_walltime_accounting;
struct MdrunOptions;
struct ObservablesHistory;
struct ReplicaExchangeParameters;
struct t_commrec;
struct t_fcdata;
struct t_forcerec;
struct t_filenm;
struct t_inputrec;
struct t_nrnb;
class t_state;

namespace gmx
{
class AccumulateGlobalsBuilder;
class BoxDeformation;
class Constraints;
class IMDOutputProvider;
class MDLogger;
class MDAtoms;
class PpForceWorkload;
class StopHandlerBuilder;

class Integrator
{
    public:
        virtual void run()    = 0;
        virtual ~Integrator() = default;
};

/*! \internal
 * \brief Element managing the current step and time
 *
 * This element keeps track of the current step and time, it should be placed in the
 * innermost part of the integrator loop. The current step and time can be accessed
 * using accessors, available using `getStepAccessor()` and `getTimeAccessor()`. Clients
 * can also register a callback to be notified when the last integrator step starts.
 */
class SimpleStepManager final : public IIntegratorElement
{
    public:
        explicit SimpleStepManager(
            real timestep, long nsteps = -1, long step = 0, real time = 0.0);

        //! IIntegratorElement functions
        ElementFunctionTypePtr registerSetup() override;
        ElementFunctionTypePtr registerRun() override;
        ElementFunctionTypePtr registerTeardown() override;

        //! Returns a function pointer allowing to access the current step number
        StepAccessorPtr getStepAccessor() const;
        //! Returns a function pointer allowing to access the current time
        TimeAccessorPtr getTimeAccessor() const;

        //! Allows clients to register a last-step callback
        void registerLastStepCallback(LastStepCallbackPtr callback);

    private:
        long step_;
        long nsteps_;
        real initialTime_;
        real time_;
        real timestep_;

        //! List of callback to be called when the last step starts
        std::vector<LastStepCallbackPtr> lastStepCallbacks_;

        //! Increments the step and time, needs to be called every step
        void increment();
        //! Called before the first step - notifies clients if first step is also last step
        void setup();
};

class PreLoopElement : public IIntegratorElement
{
    public:
        explicit PreLoopElement(gmx_wallcycle *wcycle);
        ElementFunctionTypePtr registerSetup() override;
        ElementFunctionTypePtr registerRun() override;
        ElementFunctionTypePtr registerTeardown() override;

    private:
        void run();

        gmx_wallcycle *wcycle_;
};

class PostLoopElement : public IIntegratorElement, public ILoggingSignallerClient
{
    public:
        PostLoopElement(
            StepAccessorPtr          stepAccessor,
            bool                     doVerbose,
            int                      verboseStepInterval,
            FILE                    *fplog,
            const t_inputrec        *inputrec,
            const t_commrec         *cr,
            gmx_walltime_accounting *walltime_accounting,
            gmx_wallcycle           *wcycle);

        ElementFunctionTypePtr registerSetup() override;
        ElementFunctionTypePtr registerRun() override;
        ElementFunctionTypePtr registerTeardown() override;

        LastStepCallbackPtr getLastStepCallback();

        LoggingSignallerCallbackPtr getLoggingCallback() override;
    private:
        void run();
        void nextStepIsLast();
        void thisStepIsLoggingStep();

        const bool               isMaster_;
        const bool               doDomainDecomposition_;
        const bool               doVerbose_;
        const int                verboseStepInterval_;
        bool                     isLoggingStep_;
        bool                     isFirstStep_;
        bool                     isLastStep_;

        StepAccessorPtr          stepAccessor_;

        FILE                    *fplog_;
        const t_inputrec        *inputrec_;
        const t_commrec         *cr_;
        gmx_walltime_accounting *walltime_accounting_;
        gmx_wallcycle           *wcycle_;
};

namespace legacy
{

//! Function type for integrator code.
using IntegratorFunctionType = void();

/*! \internal
 * \brief Struct to handle setting up and running the different "integrators".
 *
 * This struct is a mere aggregate of parameters to pass to evaluate an
 * energy, so that future changes to names and types of them consume
 * less time when refactoring other code.
 *
 * Aggregate initialization is used, for which the chief risk is that
 * if a member is added at the end and not all initializer lists are
 * updated, then the member will be value initialized, which will
 * typically mean initialization to zero.
 *
 * Having multiple integrators as member functions isn't a good
 * design, and we definitely only intend one to be called, but the
 * goal is to make it easy to change the names and types of members
 * without having to make identical changes in several places in the
 * code. Once many of them have become modules, we should change this
 * approach.
 *
 * Note that the presence of const reference members means that the
 * default constructor would be implicitly deleted. But we only want
 * to make one of these when we know how to initialize these members,
 * so that is perfect. To ensure this remains true even if we would
 * remove those members, we explicitly delete this constructor.
 * Other constructors, copies and moves are OK. */
struct Integrator : public gmx::Integrator
{
    //! Handles logging.
    FILE                               *fplog;
    //! Handles communication.
    t_commrec                          *cr;
    //! Coordinates multi-simulations.
    const gmx_multisim_t               *ms;
    //! Handles logging.
    const MDLogger                     &mdlog;
    //! Count of input file options.
    int                                 nfile;
    //! Content of input file options.
    const t_filenm                     *fnm;
    //! Handles writing text output.
    const gmx_output_env_t             *oenv;
    //! Contains command-line options to mdrun.
    const MdrunOptions                 &mdrunOptions;
    //! Handles virtual sites.
    gmx_vsite_t                        *vsite;
    //! Handles constraints.
    Constraints                        *constr;
    //! Handles enforced rotation.
    gmx_enfrot                         *enforcedRotation;
    //! Handles box deformation.
    BoxDeformation                     *deform;
    //! Handles writing output files.
    IMDOutputProvider                  *outputProvider;
    //! Contains user input mdp options.
    t_inputrec                         *inputrec;
    //! Full system topology.
    gmx_mtop_t                         *top_global;
    //! Helper struct for force calculations.
    t_fcdata                           *fcd;
    //! Full simulation state (only non-nullptr on master rank).
    t_state                            *state_global;
    //! History of simulation observables.
    ObservablesHistory                 *observablesHistory;
    //! Atom parameters for this domain.
    MDAtoms                            *mdAtoms;
    //! Manages flop accounting.
    t_nrnb                             *nrnb;
    //! Manages wall cycle accounting.
    gmx_wallcycle                      *wcycle;
    //! Parameters for force calculations.
    t_forcerec                         *fr;
    //! Schedule of force-calculation work each step for this task.
    PpForceWorkload                    *ppForceWorkload;
    //! Parameters for replica exchange algorihtms.
    const ReplicaExchangeParameters    &replExParams;
    //! Parameters for membrane embedding.
    gmx_membed_t                       *membed;
    //! Builds an object that will accumulates globals for modules that require that.
    AccumulateGlobalsBuilder           *accumulateGlobalsBuilder_;
    //! Manages wall time accounting.
    gmx_walltime_accounting            *walltime_accounting;
    //! Registers stop conditions
    std::unique_ptr<StopHandlerBuilder> stopHandlerBuilder;
    //! The mdp integrator field
    const int                           ei;
    //! Whether we are doing a rerun
    const bool                          doRerun;
    //! Implements the normal MD integrators.
    IntegratorFunctionType              do_md;
    //! Implements the normal MD integrators (simple version).
    IntegratorFunctionType              do_simple_md;
    //! Implements the rerun functionality.
    IntegratorFunctionType              do_rerun;
    //! Implements steepest descent EM.
    IntegratorFunctionType              do_steep;
    //! Implements conjugate gradient energy minimization
    IntegratorFunctionType              do_cg;
    //! Implements onjugate gradient energy minimization using the L-BFGS algorithm
    IntegratorFunctionType              do_lbfgs;
    //! Implements normal mode analysis
    IntegratorFunctionType              do_nm;
    //! Implements test particle insertion
    IntegratorFunctionType              do_tpi;
    //! Implements MiMiC QM/MM workflow
    IntegratorFunctionType              do_mimic;
    /*! \brief Function to run the correct IntegratorFunctionType,
     * based on the .mdp integrator field. */
    void run() override;
    //! Derived class don't work with aggregate initialization
    Integrator(
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
            bool                                doRerun
            );
};

}      // namespace legacy

}      // namespace gmx

#endif // GMX_MDRUN_INTEGRATOR_H
