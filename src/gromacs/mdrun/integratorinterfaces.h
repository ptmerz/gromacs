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
};

}      // namespace gmx

#endif //GMX_MDRUN_INTEGRATORINTERFACES_H
