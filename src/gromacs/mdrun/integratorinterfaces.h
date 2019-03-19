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
 * \brief Declares the interfaces used for the integrators
 *
 * \author Pascal Merz <pascal.merz@colorado.edu>
 * \ingroup module_mdrun
 */
#ifndef GMX_MDRUN_INTEGRATORINTERFACES_H
#define GMX_MDRUN_INTEGRATORINTERFACES_H

#include <functional>
#include <memory>

#include "gromacs/utility/real.h"

struct gmx_mdoutf;

namespace gmx
{
//! Defines the (argument-less) calls that represent setup, run and teardown of the elements
typedef std::function<void()> ElementFunctionType;
typedef std::unique_ptr<ElementFunctionType> ElementFunctionTypePtr;

/*! \internal
 * \brief Interface for all integrator elements
 *
 * All elements of the integrators need to implement this interface. It allows the interface
 * to require a pointer to the setup, run, and teardown functions, respectively. Elements can
 * return `nullptr` if no setup / run / teardown is needed.
 */
class IIntegratorElement
{
    public:
        virtual ElementFunctionTypePtr registerSetup()    = 0;
        virtual ElementFunctionTypePtr registerRun()      = 0;
        virtual ElementFunctionTypePtr registerTeardown() = 0;

        virtual ~IIntegratorElement() = default;
};

typedef std::function<void(int, int)> UpdateRunFunctionType;
typedef std::unique_ptr<UpdateRunFunctionType> UpdateRunFunctionTypePtr;
class IUpdateElement : public IIntegratorElement
{
    public:
        virtual UpdateRunFunctionTypePtr registerUpdateRun() = 0;
};

/*! \internal
 * \brief Interface for elements of the scheduled integrator
 *
 * These will decide based on a step number and time whether (and, possibly,
 * what) they will run at that specific step
 */
class ISchedulerElement
{
    public:
        virtual ElementFunctionTypePtr registerSetup()                   = 0;
        virtual ElementFunctionTypePtr scheduleRun(long step, real time) = 0;
        virtual ElementFunctionTypePtr registerTeardown()                = 0;

        virtual ~ISchedulerElement() = default;
};

/*! \internal
 * \brief Interface for signaller elements
 *
 * These elements will be ran directly by the schedule builder, and will inform
 * registered clients whether the next step will be a "special step".
 */

class ISignallerElement
{
public:
    virtual ElementFunctionTypePtr registerSetup()    = 0;
    virtual void run(long step, real time)            = 0;
    virtual ElementFunctionTypePtr registerTeardown() = 0;

    virtual ~ISignallerElement() = default;
};

//! Defines a function / function pointer which can be used to access the current step
typedef std::function<long()> StepAccessor;
typedef std::unique_ptr<StepAccessor> StepAccessorPtr;
//! Defines a function / function pointer which can be used to access the current time
typedef std::function<real()> TimeAccessor;
typedef std::unique_ptr<TimeAccessor> TimeAccessorPtr;

//! Defines a callback for the last step of the integrator run
typedef std::function<void()> LastStepCallback;
typedef std::unique_ptr<LastStepCallback> LastStepCallbackPtr;

//! Interface defining a client of the step manager needing to know when the last step happens
class ILastStepClient
{
    public:
        //! Client returns a callback function pointer
        virtual LastStepCallbackPtr getLastStepCallback() = 0;

        virtual ~ILastStepClient() = default;
};

//! Defines a callback for neighbor searching steps
typedef std::function<void()> NeighborSearchSignallerCallback;
typedef std::unique_ptr<NeighborSearchSignallerCallback> NeighborSearchSignallerCallbackPtr;

//! Interface defining a client of the neighbor search signaller
class INeighborSearchSignallerClient
{
    public:
        //! Client returns a callback function pointer
        virtual NeighborSearchSignallerCallbackPtr getNSCallback() = 0;

        virtual ~INeighborSearchSignallerClient() = default;
};

//! Defines a callback for logging steps
typedef std::function<void()> LoggingSignallerCallback;
typedef std::unique_ptr<LoggingSignallerCallback> LoggingSignallerCallbackPtr;

//! Interface defining a client of the logging step signaller
class ILoggingSignallerClient
{
    public:
        //! Client returns a callback function pointer
        virtual LoggingSignallerCallbackPtr getLoggingCallback() = 0;
};

// Defines a callback for the different energy-related steps
typedef std::function<void()> EnergySignallerCallback;
typedef std::unique_ptr<EnergySignallerCallback> EnergySignallerCallbackPtr;

/*! \brief Interface defining a client of the energy signaller
 *
 * The energy signaller signals four different energy-related special steps:
 *   - energy calculation steps
 *   - virial calculation steps
 *   - energy writing steps
 *   - free energy calculation steps
 * This interface requires to implement a callback for each of these cases,
 * but the client might chose to return `nullptr` if it is not interested in
 * a specific event.
 */
class IEnergySignallerClient
{
    public:
        virtual EnergySignallerCallbackPtr getCalculateEnergyCallback()     = 0;
        virtual EnergySignallerCallbackPtr getCalculateVirialCallback()     = 0;
        virtual EnergySignallerCallbackPtr getWriteEnergyCallback()         = 0;
        virtual EnergySignallerCallbackPtr getCalculateFreeEnergyCallback() = 0;

        virtual ~IEnergySignallerClient() = default;
};

// Defines a callback for steps in which trajectory writing is happening
typedef std::function<void()> TrajectorySignallerCallback;
typedef std::unique_ptr<TrajectorySignallerCallback> TrajectorySignallerCallbackPtr;

// Interface defining a client of the trajectory signaller
class ITrajectorySignallerClient
{
    public:
        virtual TrajectorySignallerCallbackPtr getTrajectorySignallerCallback() = 0;
};

// Define a callback which allows elements to write to the trajectory
typedef std::function<void(gmx_mdoutf* outf)> TrajectoryWriterPrePostCallback;
typedef std::unique_ptr<TrajectoryWriterPrePostCallback> TrajectoryWriterPrePostCallbackPtr;
typedef std::function<void(gmx_mdoutf* outf, long step, real time)> TrajectoryWriterCallback;
typedef std::unique_ptr<TrajectoryWriterCallback> TrajectoryWriterCallbackPtr;

/*! \brief Interface defining a client of the trajectory writer
 *
 * A trajectory writer can register 4 different functions, which will be called
 *   - at setup time
 *   - at trajectory writing steps
 *   - at energy writing steps
 *   - at teardown time
 * This interface requires to implement a callback for each of these cases,
 * but the client might chose to return `nullptr` if it is not interested in
 * a specific event. The callback must accept an argument of type `gmx_mdoutf*`
 * to hand the file pointer.
 */
class ITrajectoryWriterClient
{
    public:
        virtual TrajectoryWriterPrePostCallbackPtr registerTrajectoryWriterSetup()    = 0;
        virtual TrajectoryWriterCallbackPtr registerTrajectoryRun()            = 0;
        virtual TrajectoryWriterCallbackPtr registerEnergyRun()                = 0;
        virtual TrajectoryWriterPrePostCallbackPtr registerTrajectoryWriterTeardown() = 0;

        virtual ~ITrajectoryWriterClient() = default;
};

/*! Callback allowing domdec element to signal global reduction element
 * that bonded interactions need to be checked
 */
typedef std::function<void()> CheckNOfBondedInteractionsCallback;
typedef std::unique_ptr<CheckNOfBondedInteractionsCallback> CheckNOfBondedInteractionsCallbackPtr;
}      // namespace gmx

#endif //GMX_MDRUN_INTEGRATORINTERFACES_H
