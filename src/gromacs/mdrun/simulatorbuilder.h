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
 * \brief Declares the simulator builder for mdrun
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_mdrun
 */
#ifndef GMX_MDRUN_SIMULATORBUILDER_H
#define GMX_MDRUN_SIMULATORBUILDER_H

#include <memory>

class energyhistory_t;
struct gmx_enerdata_t;
struct gmx_enfrot;
struct gmx_mtop_t;
struct gmx_membed_t;
struct gmx_multisim_t;
struct gmx_output_env_t;
struct gmx_vsite_t;
struct gmx_wallcycle;
struct gmx_walltime_accounting;
struct ObservablesHistory;
struct pull_t;
struct ReplicaExchangeParameters;
struct t_commrec;
struct t_fcdata;
struct t_forcerec;
struct t_filenm;
struct t_inputrec;
struct t_nrnb;
struct t_swap;
class t_state;

namespace gmx
{
enum class StartingBehavior;
class BoxDeformation;
class Constraints;
class PpForceWorkload;
class IMDOutputProvider;
class ImdSession;
class MDLogger;
class MDAtoms;
class Simulator;
class StopHandlerBuilder;
struct MdrunOptions;

/*! \libinternal
 * \brief Class preparing the creation of Simulator objects
 *
 * Objects of this class build Simulator objects, which in turn are used to
 * run molecular simulations. Currently, this only has a single public
 * `build` function which takes all arguments needed to build the
 * `LegacySimulator`.
 */
class SimulatorBuilder
{
    public:
        /*! \brief Build a Simulator object based on input data
         *
         * @param[in] fplog               Log file pointer.
         * @param[in] cr                  Communication record.
         * @param[in] ms                  Multi-sim struct.
         * @param[in] mdlog               The logging interface.
         * @param[in] nfile               Count of input file options.
         * @param[in] fnm                 Content of input file options.
         * @param[in] oenv                Handles writing text output.
         * @param[in] mdrunOptions        Contains command-line options to mdrun.
         * @param[in] startingBehavior    Whether the simulation will start afresh, or restart with/without appending.
         * @param[in] vsite               Handles virtual sites.
         * @param[in] constr              Handles constraints.
         * @param[in] enforcedRotation    Handles enforced rotation.
         * @param[in] deform              Handles box deformation.
         * @param[in] outputProvider      Handles writing output files.
         * @param[in] inputrec            Contains user input mdp options.
         * @param[in] imdSession          The Interactive Molecular Dynamics session.
         * @param[in] pull_work           The pull work object.
         * @param[in] swap                The coordinate-swapping session.
         * @param[in] top_global          Full system topology.
         * @param[in] fcd                 Helper struct for force calculations.
         * @param[in] state_global        Full simulation state (only non-nullptr on master rank).
         * @param[in] observablesHistory  History of simulation observables.
         * @param[in] mdAtoms             Atom parameters for this domain.
         * @param[in] nrnb                Manages flop accounting.
         * @param[in] wcycle              Manages wall cycle accounting.
         * @param[in] fr                  Parameters for force calculations.
         * @param[in] enerd               Data for energy output.
         * @param[in] ppForceWorkload     Schedule of force-calculation work each step for this task.
         * @param[in] replExParams        Parameters for replica exchange algorihtms.
         * @param[in] membed              Parameters for membrane embedding.
         * @param[in] walltime_accounting Manages wall time accounting.
         * @param[in] stopHandlerBuilder  Registers stop conditions
         * @param[in] doRerun             Whether we're doing a rerun.
         *
         * @return                        Unique pointer to a Simulator object
         */
        std::unique_ptr<Simulator> build(
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
            bool                                doRerun);
};
}      // namespace gmx

#endif // GMX_MDRUN_SIMULATORBUILDER_SIMULATORBUILDER_H
