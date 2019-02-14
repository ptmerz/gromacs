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

/*! \internal \file
 * \brief
 * Tests to compare two integrators which are expected to be identical
 * Adapted from the tests for the mdrun -rerun functionality
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \author Pascal Merz <pascal.merz@colorado.edu>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include "config.h"

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/options/filenameoption.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/trajectory/energyframe.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/stringutil.h"

#include "testutils/mpitest.h"
#include "testutils/simulationdatabase.h"
#include "testutils/testasserts.h"

#include "energycomparison.h"
#include "energyreader.h"
#include "mdruncomparison.h"
#include "moduletest.h"
#include "trajectorycomparison.h"
#include "trajectoryreader.h"

namespace gmx
{
namespace test
{
namespace
{

//! Functor for comparing reference and test frames on particular energies to match.
class EnergyComparator
{
    public:
        //! Constructor
        explicit EnergyComparator(const EnergyTolerances &energiesToMatch)
            : energiesToMatch_(energiesToMatch) {}
        //! The functor method.
        void operator()(const EnergyFrame &reference, const EnergyFrame &test)
        {
            compareEnergyFrames(reference, test, energiesToMatch_);
        }
        //! Container of the energies to match and the tolerance required.
        const EnergyTolerances &energiesToMatch_;
};

//! Run grompp and mdrun for both sets of mdp field values
void executeIntegratorComparisonTest(
        TestFileManager        *fileManager,
        SimulationRunner       *runner,
        const std::string      &simulationName,
        int                     maxWarningsTolerated,
        const MdpFieldValues   &mdpFieldValues1,
        const MdpFieldValues   &mdpFieldValues2,
        const EnergyTolerances &energiesToMatch)
{
    // TODO At some point we should also test PME-only ranks.
    int numRanksAvailable = getNumberOfTestMpiRanks();
    if (!isNumberOfPpRanksSupported(simulationName, numRanksAvailable))
    {
        fprintf(stdout, "Test system '%s' cannot run with %d ranks.\n"
                "The supported numbers are: %s\n",
                simulationName.c_str(), numRanksAvailable,
                reportNumbersOfPpRanksSupported(simulationName).c_str());
        return;
    }

    auto integrator1TrajectoryFileName = fileManager->getTemporaryFilePath("int1.trr");
    auto integrator1EdrFileName        = fileManager->getTemporaryFilePath("int1.edr");
    auto integrator1TprFileName        = fileManager->getTemporaryFilePath("int1.tpr");
    auto integrator2TrajectoryFileName = fileManager->getTemporaryFilePath("int2.trr");
    auto integrator2EdrFileName        = fileManager->getTemporaryFilePath("int2.edr");
    auto integrator2TprFileName        = fileManager->getTemporaryFilePath("int2.tpr");

    // prepare the first .tpr file
    {
        // TODO evolve grompp to report the number of warnings issued, so
        // tests always expect the right number.
        CommandLine caller;
        caller.append("grompp");
        caller.addOption("-maxwarn", maxWarningsTolerated);
        runner->tprFileName_ = integrator1TprFileName;
        runner->useTopGroAndNdxFromDatabase(simulationName);
        auto mdpString = prepareMdpFileContents(mdpFieldValues1);
        if (mdpFieldValues1.count("userint1"))
        {
            mdpString += "userint1                = " + mdpFieldValues1.at("userint1") + "\n";
        }
        runner->useStringAsMdpFile(mdpString);
        EXPECT_EQ(0, runner->callGrompp(caller));
    }

    // do the first mdrun
    {
        runner->fullPrecisionTrajectoryFileName_ = integrator1TrajectoryFileName;
        runner->edrFileName_                     = integrator1EdrFileName;
        runner->tprFileName_                     = integrator1TprFileName;
        CommandLine integrator1Caller;
        integrator1Caller.append("mdrun");
        ASSERT_EQ(0, runner->callMdrun(integrator1Caller));
    }

    // prepare the second .tpr file
    {
        // TODO evolve grompp to report the number of warnings issued, so
        // tests always expect the right number.
        CommandLine caller;
        caller.append("grompp");
        caller.addOption("-maxwarn", maxWarningsTolerated);
        runner->tprFileName_ = integrator2TprFileName;
        runner->useTopGroAndNdxFromDatabase(simulationName);
        auto mdpString = prepareMdpFileContents(mdpFieldValues2);
        if (mdpFieldValues2.count("userint1"))
        {
            mdpString += "userint1                = " + mdpFieldValues2.at("userint1") + "\n";
        }
        runner->useStringAsMdpFile(mdpString);
        EXPECT_EQ(0, runner->callGrompp(caller));
    }

    // do the second mdrun
    {
        runner->fullPrecisionTrajectoryFileName_ = integrator2TrajectoryFileName;
        runner->edrFileName_                     = integrator2EdrFileName;
        runner->tprFileName_                     = integrator2TprFileName;
        CommandLine integrator2Caller;
        integrator2Caller.append("mdrun");
        ASSERT_EQ(0, runner->callMdrun(integrator2Caller));
    }

    // Build the functor that will compare reference and test
    // energy frames on the chosen energy fields.
    //
    // TODO It would be less code if we used a lambda for this, but either
    // clang 3.4 or libstdc++ 5.2.1 have an issue with capturing a
    // std::unordered_map
    EnergyComparator energyComparator(energiesToMatch);
    // Build the manager that will present matching pairs of frames to compare.
    //
    // TODO Here is an unnecessary copy of keys (ie. the energy field
    // names), for convenience. In the future, use a range.
    auto namesOfEnergiesToMatch = getKeys(energiesToMatch);
    FramePairManager<EnergyFrameReader, EnergyFrame>
         energyManager(
            openEnergyFileToReadFields(integrator1EdrFileName, namesOfEnergiesToMatch),
            openEnergyFileToReadFields(integrator2EdrFileName, namesOfEnergiesToMatch));
    // Compare the energy frames.
    energyManager.compareAllFramePairs(energyComparator);

    // Specify how trajectory frame matching must work.
    TrajectoryFrameMatchSettings trajectoryMatchSettings {
        true, true, true, true, true, true
    };
    /* Specify the default expected tolerances for trajectory
     * components for all simulation systems. */
    TrajectoryTolerances trajectoryTolerances {
        defaultRealTolerance(),                                               // box
        relativeToleranceAsFloatingPoint(1.0, 1.0e-3),                        // positions
        defaultRealTolerance(),                                               // velocities
        relativeToleranceAsFloatingPoint(100.0, GMX_DOUBLE ? 1.0e-7 : 1.0e-5) // forces
    };

    // Build the functor that will compare reference and test
    // trajectory frames in the chosen way.
    auto trajectoryComparator = [&trajectoryMatchSettings, &trajectoryTolerances](const TrajectoryFrame &reference, const TrajectoryFrame &test)
        {
            compareTrajectoryFrames(reference, test, trajectoryMatchSettings, trajectoryTolerances);
        };
    // Build the manager that will present matching pairs of frames to compare
    FramePairManager<TrajectoryFrameReader, TrajectoryFrame>
    trajectoryManager(std::make_unique<TrajectoryFrameReader>(integrator1TrajectoryFileName),
                      std::make_unique<TrajectoryFrameReader>(integrator2TrajectoryFileName));
    // Compare the trajectory frames.
    trajectoryManager.compareAllFramePairs(trajectoryComparator);
}

/*! \brief Test fixture base for two identical integrators
 *
 * This test ensures that two integrator code paths (called via different mdp
 * options) yield identical coordinate, velocity, box, force and energy
 * trajectories, up to some (arbitrary) precision.
 *
 * These tests are useful to check that re-implementations of existing integrators
 * are correct, and that different code paths expected to yield identical results
 * are equivalent.
 *
 * TODO: Rethink the precisions.
 * TODO: The current mdp input might not be enough soon - we might need more than
 *       only simulation name and integrator type to identify integrators.
 */
class IntegratorComparisonTest :
    public MdrunTestFixture,
    public ::testing::WithParamInterface < std::tuple < std::string, std::string>>
{
};

TEST_P(IntegratorComparisonTest, WithinTolerances)
{
    auto params         = GetParam();
    auto simulationName = std::get<0>(params);
    auto integrator     = std::get<1>(params);
    SCOPED_TRACE(formatString("Comparing two simulations of '%s' "
                              "with integrator '%s'",
                              simulationName.c_str(), integrator.c_str()));

    auto mdpFieldValues1 = prepareMdpFieldValues(simulationName.c_str(),
                                                 integrator.c_str(),
                                                 "no", "no");
    auto mdpFieldValues2 = prepareMdpFieldValues(simulationName.c_str(),
                                                 integrator.c_str(),
                                                 "no", "no");
    // Add the magic value hack to use alternative code path
    using MdpField = MdpFieldValues::value_type;
    mdpFieldValues2.insert(MdpField("userint1", "99887766"));

    EnergyTolerances energiesToMatch
    {{
         {
             interaction_function[F_EPOT].longname,
             relativeToleranceAsPrecisionDependentUlp(10.0, 24, 40)
         },
     }};

    int numWarningsToTolerate = 0;
    executeIntegratorComparisonTest(
            &fileManager_, &runner_,
            simulationName, numWarningsToTolerate,
            mdpFieldValues1, mdpFieldValues2,
            energiesToMatch);
}

// TODO The time for OpenCL kernel compilation means these tests time
// out. Once that compilation is cached for the whole process, these
// tests can run in such configurations.
#if GMX_GPU != GMX_GPU_OPENCL
INSTANTIATE_TEST_CASE_P(IntegratorsAreEquivalent, IntegratorComparisonTest,
                            ::testing::Combine(::testing::Values("argon12", "tip3p5"),
                                                   ::testing::Values("md", "md-vv")));
#else
INSTANTIATE_TEST_CASE_P(DISABLED_IntegratorsAreEquivalent, IntegratorComparisonTest,
                            ::testing::Combine(::testing::Values("argon12", "tip3p5"),
                                                   ::testing::Values("md", "md-vv")));
#endif

} // namespace
} // namespace test
} // namespace gmx
