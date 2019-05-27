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
 * \brief Defines a simplified dispatch function for the .mdp integrator field.
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_mdrun
 */
#include "gmxpre.h"

#include "simplesimulator.h"

#include "gromacs/mdlib/stophandler.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/utility/exceptions.h"

namespace gmx
{
//! Constructor
SimpleSimulator::SimpleSimulator(
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
        bool                                doRerun) :
    fplog(fplog),
    cr(cr),
    ms(ms),
    mdlog(mdlog),
    nfile(nfile),
    fnm(fnm),
    oenv(oenv),
    mdrunOptions(mdrunOptions),
    startingBehavior(startingBehavior),
    vsite(vsite),
    constr(constr),
    enforcedRotation(enforcedRotation),
    deform(deform),
    outputProvider(outputProvider),
    inputrec(inputrec),
    imdSession(imdSession),
    pull_work(pull_work),
    swap(swap),
    top_global(top_global),
    fcd(fcd),
    state_global(state_global),
    observablesHistory(observablesHistory),
    mdAtoms(mdAtoms),
    nrnb(nrnb),
    wcycle(wcycle),
    fr(fr),
    enerd(enerd),
    ppForceWorkload(ppForceWorkload),
    replExParams(replExParams),
    membed(membed),
    walltime_accounting(walltime_accounting),
    stopHandlerBuilder(std::move(stopHandlerBuilder)),
    doRerun(doRerun)
{}

//! \brief Run the correct integrator function.
void SimpleSimulator::run()
{
    switch (inputrec->eI)
    {
        case eiMD:
        case eiBD:
        case eiSD1:
        case eiVV:
        case eiVVAK:
            if (!EI_DYNAMICS(inputrec->eI))
            {
                GMX_THROW(APIError("do_md integrator would be called for a non-dynamical integrator"));
            }
            if (!doRerun)
            {
                do_simplemd();
                break;
            }
        default:
            GMX_THROW(APIError("Selected integrator is not compatible with modular simulator."));
    }
}
}  // namespace gmx
