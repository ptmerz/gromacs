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
/*! \internal \file
 *
 * \brief Implements a simplified version of the simulator for normal MD simulations
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_mdrun
 */
#include "gmxpre.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <memory>

#include "gromacs/domdec/collect.h"
#include "gromacs/domdec/dlbtiming.h"
#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_network.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/domdec/mdsetup.h"
#include "gromacs/domdec/partition.h"
#include "gromacs/ewald/pme.h"
#include "gromacs/ewald/pme_load_balancing.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/listed_forces/manage_threading.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/utilities.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/compute_io.h"
#include "gromacs/mdlib/constr.h"
#include "gromacs/mdlib/ebin.h"
#include "gromacs/mdlib/enerdata_utils.h"
#include "gromacs/mdlib/energyoutput.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdlib/forcerec.h"
#include "gromacs/mdlib/md_support.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdlib/mdoutf.h"
#include "gromacs/mdlib/resethandler.h"
#include "gromacs/mdlib/sighandler.h"
#include "gromacs/mdlib/simulationsignal.h"
#include "gromacs/mdlib/stat.h"
#include "gromacs/mdlib/stophandler.h"
#include "gromacs/mdlib/tgroup.h"
#include "gromacs/mdlib/trajectory_writing.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdrunutility/handlerestart.h"
#include "gromacs/mdrunutility/printtime.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/energyhistory.h"
#include "gromacs/mdtypes/fcdata.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/mdrunoptions.h"
#include "gromacs/mdtypes/observableshistory.h"
#include "gromacs/mdtypes/pullhistory.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/pbcutil/mshift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pulling/output.h"
#include "gromacs/pulling/pull.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/timing/walltime_accounting.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/smalloc.h"

#include "shellfc.h"
#include "simplesimulator.h"

#if GMX_FAHCORE
#include "corewrap.h"
#endif

using gmx::SimulationSignaller;

void gmx::SimpleSimulator::do_simplemd()
{
    // TODO Historically, the EM and MD "integrators" used different
    // names for the t_inputrec *parameter, but these must have the
    // same name, now that it's a member of a struct. We use this ir
    // alias to avoid a large ripple of nearly useless changes.
    // t_inputrec is being replaced by IMdpOptionsProvider, so this
    // will go away eventually.
    t_inputrec             *ir   = inputrec;
    int64_t                 step, step_rel;
    double                  t, t0 = ir->init_t;
    gmx_bool                bGStat, bCalcVir, bCalcEnerStep, bCalcEner;
    gmx_bool                bNS, bStopCM,
                            bFirstStep, bInitStep, bLastStep = FALSE;
    gmx_bool                do_ene, do_log, do_verbose;
    int                     force_flags, cglo_flags;
    tensor                  force_vir = {{0}}, shake_vir = {{0}}, total_vir = {{0}}, pres = {{0}};
    int                     i, m;
    rvec                    mu_tot;
    matrix                  M;
    gmx_localtop_t          top;
    PaddedVector<gmx::RVec> f {};
    gmx_global_stat_t       gstat;
    t_graph                *graph = nullptr;
    gmx_shellfc_t          *shellfc;
    gmx_bool                bSumEkinhOld;
    gmx_bool                bTemp, bPres;
    real                    dvdl_constr;
    matrix                  lastbox;
    double                  cycles;
    t_extmass               MassQ;
    char                    sbuf[STEPSTRSIZE], sbuf2[STEPSTRSIZE];

    /* PME load balancing data for GPU kernels */
    gmx_bool              bPMETune         = FALSE;
    gmx_bool              bPMETunePrinting = FALSE;

    /* Domain decomposition could incorrectly miss a bonded
       interaction, but checking for that requires a global
       communication stage, which does not otherwise happen in DD
       code. So we do that alongside the first global energy reduction
       after a new DD is made. These variables handle whether the
       check happens, and the result it returns. */
    bool              shouldCheckNumberOfBondedInteractions = false;
    int               totalNumberOfBondedInteractions       = -1;

    SimulationSignals signals;
    // Most global communnication stages don't propagate mdrun
    // signals, and will use this object to achieve that.
    SimulationSignaller nullSignaller(nullptr, nullptr, nullptr, false, false);

    if (!mdrunOptions.writeConfout)
    {
        // This is on by default, and the main known use case for
        // turning it off is for convenience in benchmarking, which is
        // something that should not show up in the general user
        // interface.
        GMX_LOG(mdlog.info).asParagraph().
            appendText("The -noconfout functionality is deprecated, and may be removed in a future version.");
    }

    const bool bRerunMD      = false;

    int        nstglobalcomm = computeGlobalCommunicationPeriod(mdlog, ir, cr);

    SimulationGroups                  *groups = &top_global->groups;

    Update upd(ir, deform);

    pleaseCiteCouplingAlgorithms(fplog, *ir);

    init_nrnb(nrnb);
    gmx_mdoutf       *outf = init_mdoutf(fplog, nfile, fnm, mdrunOptions, cr, outputProvider, ir, top_global, oenv, wcycle,
                                         StartingBehavior::NewSimulation);
    gmx::EnergyOutput energyOutput(mdoutf_get_fp_ene(outf), top_global, ir, pull_work, mdoutf_get_fp_dhdl(outf), false);

    /* Kinetic energy data */
    std::unique_ptr<gmx_ekindata_t> eKinData = std::make_unique<gmx_ekindata_t>();
    gmx_ekindata_t                 *ekind    = eKinData.get();
    init_ekindata(fplog, top_global, &(ir->opts), ekind);
    /* Copy the cos acceleration to the groups struct */
    ekind->cosacc.cos_accel = ir->cos_accel;

    gstat = global_stat_init(ir);

    /* Check for polarizable models and flexible constraints */
    shellfc = init_shell_flexcon(fplog,
                                 top_global, constr ? constr->numFlexibleConstraints() : 0,
                                 ir->nstcalcenergy, DOMAINDECOMP(cr));

    {
        double io = compute_io(ir, top_global->natoms, *groups, energyOutput.numEnergyTerms(), 1);
        if ((io > 2000) && MASTER(cr))
        {
            fprintf(stderr,
                    "\nWARNING: This run will generate roughly %.0f Mb of data\n\n",
                    io);
        }
    }

    // Local state only becomes valid now.
    std::unique_ptr<t_state> stateInstance;
    t_state *                state;

    if (DOMAINDECOMP(cr))
    {
        dd_init_local_top(*top_global, &top);

        stateInstance = std::make_unique<t_state>();
        state         = stateInstance.get();
        if (fr->nbv->useGpu())
        {
            changePinningPolicy(&state->x, gmx::PinningPolicy::PinnedIfSupported);
        }
        dd_init_local_state(cr->dd, state_global, state);

        /* Distribute the charge groups over the nodes from the master node */
        dd_partition_system(fplog, mdlog, ir->init_step, cr, TRUE, 1,
                            state_global, *top_global, ir, imdSession,
                            pull_work,
                            state, &f, mdAtoms, &top, fr,
                            vsite, constr,
                            nrnb, nullptr, FALSE);
        shouldCheckNumberOfBondedInteractions = true;
        upd.setNumAtoms(state->natoms);
    }
    else
    {
        state_change_natoms(state_global, state_global->natoms);
        f.resizeWithPadding(state_global->natoms);
        /* Copy the pointer to the global state */
        state = state_global;

        /* Generate and initialize new topology */
        mdAlgorithmsSetupAtomData(cr, ir, *top_global, &top, fr,
                                  &graph, mdAtoms, constr, vsite, shellfc);

        upd.setNumAtoms(state->natoms);
    }

    auto mdatoms = mdAtoms->mdatoms();

    // NOTE: The global state is no longer used at this point.
    // But state_global is still used as temporary storage space for writing
    // the global state to file and potentially for replica exchange.
    // (Global topology should persist.)

    if (MASTER(cr))
    {
        if (!observablesHistory->energyHistory)
        {
            observablesHistory->energyHistory = std::make_unique<energyhistory_t>();
        }
        if (!observablesHistory->pullHistory)
        {
            observablesHistory->pullHistory = std::make_unique<PullHistory>();
        }
        /* Set the initial energy history */
        energyOutput.fillEnergyHistory(observablesHistory->energyHistory.get());
    }

    // Disable functionality
    const auto isNewSimulation = false;
    preparePrevStepPullCom(ir, pull_work, mdatoms, state, state_global, cr,
                           !isNewSimulation);

    /* PME tuning is only supported in the Verlet scheme, with PME for
     * Coulomb. It is not supported with only LJ PME. */
    bPMETune = (mdrunOptions.tunePme && EEL_PME(fr->ic->eeltype) &&
                !mdrunOptions.reproducible && ir->cutoff_scheme != ecutsGROUP);

    pme_load_balancing_t *pme_loadbal      = nullptr;
    if (bPMETune)
    {
        pme_loadbal_init(&pme_loadbal, cr, mdlog, *ir, state->box,
                         *fr->ic, *fr->nbv, fr->pmedata, fr->nbv->useGpu(),
                         &bPMETunePrinting);
    }

    if (!ir->bContinuation)
    {
        if (state->flags & (1 << estV))
        {
            auto v = makeArrayRef(state->v);
            /* Set the velocities of vsites, shells and frozen atoms to zero */
            for (i = 0; i < mdatoms->homenr; i++)
            {
                if (mdatoms->ptype[i] == eptVSite ||
                    mdatoms->ptype[i] == eptShell)
                {
                    clear_rvec(v[i]);
                }
                else if (mdatoms->cFREEZE)
                {
                    for (m = 0; m < DIM; m++)
                    {
                        if (ir->opts.nFreeze[mdatoms->cFREEZE[i]][m])
                        {
                            v[i][m] = 0;
                        }
                    }
                }
            }
        }

        if (constr)
        {
            /* Constrain the initial coordinates and velocities */
            do_constrain_first(fplog, constr, ir, mdatoms, state);
        }
    }

    /* Be REALLY careful about what flags you set here. You CANNOT assume
     * this is the first step, since we might be restarting from a checkpoint,
     * and in that case we should not do any modifications to the state.
     */
    bStopCM = (ir->comm_mode != ecmNO && !ir->bContinuation);

    cglo_flags = (CGLO_INITIALIZATION | CGLO_TEMPERATURE | CGLO_GSTAT
                  | (EI_VV(ir->eI) ? CGLO_PRESSURE : 0)
                  | (EI_VV(ir->eI) ? CGLO_CONSTRAINT : 0));

    bSumEkinhOld = FALSE;

    t_vcm vcm(top_global->groups, *ir);
    reportComRemovalInfo(fplog, vcm);

    /* To minimize communication, compute_globals computes the COM velocity
     * and the kinetic energy for the velocities without COM motion removed.
     * Thus to get the kinetic energy without the COM contribution, we need
     * to call compute_globals twice.
     */
    for (int cgloIteration = 0; cgloIteration < (bStopCM ? 2 : 1); cgloIteration++)
    {
        int cglo_flags_iteration = cglo_flags;
        if (bStopCM && cgloIteration == 0)
        {
            cglo_flags_iteration |= CGLO_STOPCM;
            cglo_flags_iteration &= ~CGLO_TEMPERATURE;
        }
        compute_globals(fplog, gstat, cr, ir, fr, ekind, state, mdatoms, nrnb, &vcm,
                        nullptr, enerd, force_vir, shake_vir, total_vir, pres, mu_tot,
                        constr, &nullSignaller, state->box,
                        &totalNumberOfBondedInteractions, &bSumEkinhOld, cglo_flags_iteration
                        | (shouldCheckNumberOfBondedInteractions ? CGLO_CHECK_NUMBER_OF_BONDED_INTERACTIONS : 0));
    }
    checkNumberOfBondedInteractions(mdlog, cr, totalNumberOfBondedInteractions,
                                    top_global, &top, state,
                                    &shouldCheckNumberOfBondedInteractions);

    /* Calculate the initial half step temperature, and save the ekinh_old */
    for (i = 0; (i < ir->opts.ngtc); i++)
    {
        copy_mat(ekind->tcstat[i].ekinh, ekind->tcstat[i].ekinh_old);
    }

    if (MASTER(cr))
    {
        if (!ir->bContinuation)
        {
            if (constr && ir->eConstrAlg == econtLINCS)
            {
                fprintf(fplog,
                        "RMS relative constraint deviation after constraining: %.2e\n",
                        constr->rmsd());
            }
            if (EI_STATE_VELOCITY(ir->eI))
            {
                real temp = enerd->term[F_TEMP];
                if (ir->eI != eiVV)
                {
                    /* Result of Ekin averaged over velocities of -half
                     * and +half step, while we only have -half step here.
                     */
                    temp *= 2;
                }
                fprintf(fplog, "Initial temperature: %g K\n", temp);
            }
        }

        char tbuf[20];
        fprintf(stderr, "starting mdrun '%s'\n",
                *(top_global->name));
        if (ir->nsteps >= 0)
        {
            sprintf(tbuf, "%8.1f", (ir->init_step+ir->nsteps)*ir->delta_t);
        }
        else
        {
            sprintf(tbuf, "%s", "infinite");
        }
        if (ir->init_step > 0)
        {
            fprintf(stderr, "%s steps, %s ps (continuing from step %s, %8.1f ps).\n",
                    gmx_step_str(ir->init_step+ir->nsteps, sbuf), tbuf,
                    gmx_step_str(ir->init_step, sbuf2),
                    ir->init_step*ir->delta_t);
        }
        else
        {
            fprintf(stderr, "%s steps, %s ps.\n",
                    gmx_step_str(ir->nsteps, sbuf), tbuf);
        }
        fprintf(fplog, "\n");
    }

    walltime_accounting_start_time(walltime_accounting);
    wallcycle_start(wcycle, ewcRUN);
    print_start(fplog, cr, walltime_accounting, "mdrun");

#if GMX_FAHCORE
    /* safest point to do file checkpointing is here.  More general point would be immediately before integrator call */
    int chkpt_ret = fcCheckPointParallel( cr->nodeid,
                                          NULL, 0);
    if (chkpt_ret == 0)
    {
        gmx_fatal( 3, __FILE__, __LINE__, "Checkpoint error on step %d\n", 0 );
    }
#endif

    /***********************************************************
     *
     *             Loop over MD steps
     *
     ************************************************************/

    bFirstStep       = TRUE;
    /* Skip the first Nose-Hoover integration when we get the state from tpx */
    bInitStep        = TRUE;
    bSumEkinhOld     = FALSE;

    const bool simulationsShareState = false;
    int        nstSignalComm         = nstglobalcomm;

    auto       stopHandler = stopHandlerBuilder->getStopHandlerMD(
                compat::not_null<SimulationSignal*>(&signals[eglsSTOPCOND]), simulationsShareState,
                MASTER(cr), ir->nstlist, mdrunOptions.reproducible, nstSignalComm,
                mdrunOptions.maximumHoursToRun, ir->nstlist == 0, fplog, step, bNS, walltime_accounting);

    const bool resetCountersIsLocal = true;
    auto       resetHandler         = std::make_unique<ResetHandler>(
                compat::make_not_null<SimulationSignal*>(&signals[eglsRESETCOUNTERS]), !resetCountersIsLocal,
                ir->nsteps, MASTER(cr), mdrunOptions.timingOptions.resetHalfway,
                mdrunOptions.maximumHoursToRun, mdlog, wcycle, walltime_accounting);

    const DDBalanceRegionHandler ddBalanceRegionHandler(cr);

    step     = ir->init_step;
    step_rel = 0;

    /* and stop now if we should */
    bLastStep = (bLastStep || (ir->nsteps >= 0 && step_rel > ir->nsteps));
    while (!bLastStep)
    {
        /* Determine whether or not to do Neighbour Searching */
        bNS = (bFirstStep || (ir->nstlist > 0  && step % ir->nstlist == 0));

        if (bPMETune && bNS && !bFirstStep)
        {
            /* PME grid + cut-off optimization with GPUs or PME nodes */
            pme_loadbal_do(pme_loadbal, cr,
                           (mdrunOptions.verbose && MASTER(cr)) ? stderr : nullptr,
                           fplog, mdlog,
                           *ir, fr, *state,
                           wcycle,
                           step, step_rel,
                           &bPMETunePrinting);
        }

        wallcycle_start(wcycle, ewcSTEP);

        bLastStep = (step_rel == ir->nsteps);
        t         = t0 + step*ir->delta_t;

        /* Stop Center of Mass motion */
        bStopCM = (ir->comm_mode != ecmNO && do_per_step(step, ir->nstcomm));

        bLastStep = bLastStep || stopHandler->stoppingAfterCurrentStep(bNS);

        /* do_log triggers energy and virial calculation. Because this leads
         * to different code paths, forces can be different. Thus for exact
         * continuation we should avoid extra log output.
         * Note that the || bLastStep can result in non-exact continuation
         * beyond the last step. But we don't consider that to be an issue.
         */
        do_log     = do_per_step(step, ir->nstlog) || bFirstStep || bLastStep;
        do_verbose = mdrunOptions.verbose &&
            (step % mdrunOptions.verboseStepPrintInterval == 0 || bFirstStep || bLastStep);

        if (bNS && !(bFirstStep && ir->bContinuation))
        {
            if (DOMAINDECOMP(cr))
            {
                const bool bMasterState = false;
                /* Repartition the domain decomposition */
                dd_partition_system(fplog, mdlog, step, cr,
                                    bMasterState, nstglobalcomm,
                                    state_global, *top_global, ir, imdSession,
                                    pull_work,
                                    state, &f, mdAtoms, &top, fr,
                                    vsite, constr,
                                    nrnb, wcycle,
                                    do_verbose && !bPMETunePrinting);
                shouldCheckNumberOfBondedInteractions = true;
                upd.setNumAtoms(state->natoms);
            }
        }

        if (MASTER(cr) && do_log)
        {
            energyOutput.printHeader(fplog, step, t); /* can we improve the information printed here? */
        }

        clear_mat(force_vir);

        /* Determine the energy and pressure:
         * at nstcalcenergy steps and at energy output steps (set below).
         */
        bCalcEnerStep = do_per_step(step, ir->nstcalcenergy);
        bCalcVir      = bCalcEnerStep;
        bCalcEner     = bCalcEnerStep;

        do_ene = (do_per_step(step, ir->nstenergy) || bLastStep);

        if (do_ene || do_log)
        {
            bCalcVir  = TRUE;
            bCalcEner = TRUE;
        }

        /* Do we need global communication ? */
        bGStat = (bCalcVir || bCalcEner || bStopCM ||
                  do_per_step(step, nstglobalcomm));

        force_flags = (GMX_FORCE_STATECHANGED |
                       GMX_FORCE_ALLFORCES |
                       (bCalcVir ? GMX_FORCE_VIRIAL : 0) |
                       (bCalcEner ? GMX_FORCE_ENERGY : 0));

        if (shellfc)
        {
            /* Now is the time to relax the shells */
            relax_shell_flexcon(fplog, cr, ms, mdrunOptions.verbose,
                                enforcedRotation, step,
                                ir, imdSession, pull_work, bNS, force_flags, &top,
                                constr, enerd, fcd,
                                state, f.arrayRefWithPadding(), force_vir, mdatoms,
                                nrnb, wcycle, graph,
                                shellfc, fr, ppForceWorkload, t, mu_tot,
                                vsite,
                                ddBalanceRegionHandler);
        }
        else
        {
            // Disabled functionality
            Awh       *awh = nullptr;
            gmx_edsam *ed  = nullptr;

            /* The coordinates (x) are shifted (to get whole molecules)
             * in do_force.
             * This is parallellized as well, and does communication too.
             * Check comments in sim_util.c
             */
            do_force(fplog, cr, ms, ir, awh, enforcedRotation, imdSession,
                     pull_work,
                     step, nrnb, wcycle, &top,
                     state->box, state->x.arrayRefWithPadding(), &state->hist,
                     f.arrayRefWithPadding(), force_vir, mdatoms, enerd, fcd,
                     state->lambda, graph,
                     fr, ppForceWorkload, vsite, mu_tot, t, ed,
                     (bNS ? GMX_FORCE_NS : 0) | force_flags,
                     ddBalanceRegionHandler);
        }

        if (ir->eI == eiVV)
        /*  ############### START FIRST UPDATE HALF-STEP FOR VV METHODS############### */
        {
            rvec *vbuf = nullptr;

            wallcycle_start(wcycle, ewcUPDATE);
            if (bInitStep)
            {
                /* if using velocity verlet with full time step Ekin,
                 * take the first half step only to compute the
                 * virial for the first step. From there,
                 * revert back to the initial coordinates
                 * so that the input is actually the initial step.
                 */
                snew(vbuf, state->natoms);
                copy_rvecn(state->v.rvec_array(), vbuf, 0, state->natoms); /* should make this better for parallelizing? */
            }

            update_coords(step, ir, mdatoms, state, f.arrayRefWithPadding(), fcd,
                          ekind, M, &upd, etrtVELOCITY1,
                          cr, constr);

            wallcycle_stop(wcycle, ewcUPDATE);
            constrain_velocities(step, nullptr,
                                 state,
                                 shake_vir,
                                 constr,
                                 bCalcVir, do_log, do_ene);
            wallcycle_start(wcycle, ewcUPDATE);
            /* if VV, compute the pressure and constraints */
            /* For VV2, we strictly only need this if using pressure
             * control, but we really would like to have accurate pressures
             * printed out.
             * Think about ways around this in the future?
             * For now, keep this choice in comments.
             */
            /*bPres = (ir->eI==eiVV || inputrecNptTrotter(ir)); */
            /*bTemp = ((ir->eI==eiVV &&(!bInitStep)) || (ir->eI==eiVVAK && inputrecNptTrotter(ir)));*/
            bPres = TRUE;
            bTemp = (!bInitStep);
            /* for vv, the first half of the integration actually corresponds to the previous step.
               So we need information from the last step in the first half of the integration */
            if (bGStat || do_per_step(step-1, nstglobalcomm))
            {
                wallcycle_stop(wcycle, ewcUPDATE);
                compute_globals(fplog, gstat, cr, ir, fr, ekind, state, mdatoms, nrnb, &vcm,
                                wcycle, enerd, force_vir, shake_vir, total_vir, pres, mu_tot,
                                constr, &nullSignaller, state->box,
                                &totalNumberOfBondedInteractions, &bSumEkinhOld,
                                (bGStat ? CGLO_GSTAT : 0)
                                | (bCalcEner ? CGLO_ENERGY : 0)
                                | (bTemp ? CGLO_TEMPERATURE : 0)
                                | (bPres ? CGLO_PRESSURE : 0)
                                | (bPres ? CGLO_CONSTRAINT : 0)
                                | (bStopCM ? CGLO_STOPCM : 0)
                                | (shouldCheckNumberOfBondedInteractions ? CGLO_CHECK_NUMBER_OF_BONDED_INTERACTIONS : 0)
                                | CGLO_SCALEEKIN
                                );
                /* explanation of above:
                   a) We compute Ekin at the full time step
                   if 1) we are using the AveVel Ekin, and it's not the
                   initial step, or 2) if we are using AveEkin, but need the full
                   time step kinetic energy for the pressure (always true now, since we want accurate statistics).
                   b) If we are using EkinAveEkin for the kinetic energy for the temperature control, we still feed in
                   EkinAveVel because it's needed for the pressure */
                checkNumberOfBondedInteractions(mdlog, cr, totalNumberOfBondedInteractions,
                                                top_global, &top, state,
                                                &shouldCheckNumberOfBondedInteractions);
                wallcycle_start(wcycle, ewcUPDATE);
            }
            /* if it's the initial step, we performed this first step just to get the constraint virial */
            if (bInitStep)
            {
                copy_rvecn(vbuf, state->v.rvec_array(), 0, state->natoms);
                sfree(vbuf);
            }
            wallcycle_stop(wcycle, ewcUPDATE);
        }

        /* ########  END FIRST UPDATE STEP  ############## */
        /* ########  If doing VV, we now have v(dt) ###### */

        // Disable functionality
        const auto checkpointingStep = false;

        /* Now we have the energies and forces corresponding to the
         * coordinates at time t. We must output all of this before
         * the update.
         */
        do_md_trajectory_writing(fplog, cr, nfile, fnm, step, step_rel, t,
                                 ir, state, state_global, observablesHistory,
                                 top_global, fr,
                                 outf, energyOutput, ekind, f,
                                 checkpointingStep,
                                 bRerunMD, bLastStep,
                                 mdrunOptions.writeConfout,
                                 bSumEkinhOld);

        stopHandler->setSignal();
        resetHandler->setSignal(walltime_accounting);

        /* #########   START SECOND UPDATE STEP ################# */

        /* Box is changed in update() when we do pressure coupling,
         * but we should still use the old box for energy corrections and when
         * writing it to the energy file, so it matches the trajectory files for
         * the same timestep above. Make a copy in a separate array.
         */
        copy_mat(state->box, lastbox);

        dvdl_constr = 0;

        wallcycle_start(wcycle, ewcUPDATE);
        if (EI_VV(ir->eI))
        {
            /* velocity half-step update */
            update_coords(step, ir, mdatoms, state, f.arrayRefWithPadding(), fcd,
                          ekind, M, &upd, etrtVELOCITY2,
                          cr, constr);
        }

        update_coords(step, ir, mdatoms, state, f.arrayRefWithPadding(), fcd,
                      ekind, M, &upd, etrtPOSITION, cr, constr);
        wallcycle_stop(wcycle, ewcUPDATE);

        constrain_coordinates(step, &dvdl_constr, state,
                              shake_vir,
                              &upd, constr,
                              bCalcVir, do_log, do_ene);
        update_sd_second_half(step, &dvdl_constr, ir, mdatoms, state,
                              cr, nrnb, wcycle, &upd, constr, do_log, do_ene);
        finish_update(ir, mdatoms,
                      state, graph,
                      nrnb, wcycle, &upd, constr);

        if (ir->bPull && ir->pull->bSetPbcRefToPrevStepCOM)
        {
            updatePrevStepPullCom(pull_work, state);
        }

        /* ############## IF NOT VV, Calculate globals HERE  ############ */
        /* With Leap-Frog we can skip compute_globals at
         * non-communication steps, but we need to calculate
         * the kinetic energy one step before communication.
         */
        {
            // Disable functionality
            const bool doInterSimSignal = false;

            if (bGStat || (!EI_VV(ir->eI) && do_per_step(step+1, nstglobalcomm)))
            {
                // Since we're already communicating at this step, we
                // can propagate intra-simulation signals. Note that
                // check_nstglobalcomm has the responsibility for
                // choosing the value of nstglobalcomm that is one way
                // bGStat becomes true, so we can't get into a
                // situation where e.g. checkpointing can't be
                // signalled.
                bool                doIntraSimSignal = true;
                SimulationSignaller signaller(&signals, cr, ms, doInterSimSignal, doIntraSimSignal);

                compute_globals(fplog, gstat, cr, ir, fr, ekind, state, mdatoms, nrnb, &vcm,
                                wcycle, enerd, force_vir, shake_vir, total_vir, pres, mu_tot,
                                constr, &signaller,
                                lastbox,
                                &totalNumberOfBondedInteractions, &bSumEkinhOld,
                                (bGStat ? CGLO_GSTAT : 0)
                                | (!EI_VV(ir->eI) && bCalcEner ? CGLO_ENERGY : 0)
                                | (!EI_VV(ir->eI) && bStopCM ? CGLO_STOPCM : 0)
                                | (!EI_VV(ir->eI) ? CGLO_TEMPERATURE : 0)
                                | (!EI_VV(ir->eI) ? CGLO_PRESSURE : 0)
                                | CGLO_CONSTRAINT
                                | (shouldCheckNumberOfBondedInteractions ? CGLO_CHECK_NUMBER_OF_BONDED_INTERACTIONS : 0)
                                );
                checkNumberOfBondedInteractions(mdlog, cr, totalNumberOfBondedInteractions,
                                                top_global, &top, state,
                                                &shouldCheckNumberOfBondedInteractions);
            }
        }

        /* ################# END UPDATE STEP 2 ################# */
        /* #### We now have r(t+dt) and v(t+dt/2)  ############# */

        /* The coordinates (x) were unshifted in update */
        if (!bGStat)
        {
            /* We will not sum ekinh_old,
             * so signal that we still have to do it.
             */
            bSumEkinhOld = TRUE;
        }

        if (bCalcEner)
        {
            /* #########  BEGIN PREPARING EDR OUTPUT  ###########  */
            enerd->term[F_ETOT] = enerd->term[F_EPOT] + enerd->term[F_EKIN];
            /* #########  END PREPARING EDR OUTPUT  ###########  */
        }

        /* Output stuff */
        if (MASTER(cr))
        {
            if (bCalcEner)
            {
                // Disable functionality
                const bool bDoDHDL = false;
                energyOutput.addDataAtEnergyStep(bDoDHDL, bCalcEnerStep,
                                                 t, mdatoms->tmass, enerd, state,
                                                 ir->fepvals, ir->expandedvals, lastbox,
                                                 shake_vir, force_vir, total_vir, pres,
                                                 ekind, mu_tot, constr);
            }
            else
            {
                energyOutput.recordNonEnergyStep();
            }

            gmx_bool do_dr  = do_per_step(step, ir->nstdisreout);
            gmx_bool do_or  = do_per_step(step, ir->nstorireout);

            energyOutput.printAnnealingTemperatures(do_log ? fplog : nullptr, groups, &(ir->opts));
            // Disable functionality
            Awh *awh = nullptr;
            energyOutput.printStepToEnergyFile(mdoutf_get_fp_ene(outf), do_ene, do_dr, do_or,
                                               do_log ? fplog : nullptr,
                                               step, t,
                                               fcd, awh);

            if (ir->bPull)
            {
                pull_print_output(pull_work, step, t);
            }

            if (do_per_step(step, ir->nstlog))
            {
                if (fflush(fplog) != 0)
                {
                    gmx_fatal(FARGS, "Cannot flush logfile - maybe you are out of disk space?");
                }
            }
        }
        /* Print the remaining wall clock time for the run */
        if (MASTER(cr) &&
            (do_verbose || gmx_got_usr_signal()) &&
            !bPMETunePrinting)
        {
            if (shellfc)
            {
                fprintf(stderr, "\n");
            }
            print_time(stderr, walltime_accounting, step, ir, cr);
        }

        bFirstStep             = FALSE;
        bInitStep              = FALSE;

        cycles = wallcycle_stop(wcycle, ewcSTEP);
        if (DOMAINDECOMP(cr) && wcycle)
        {
            dd_cycles_add(cr->dd, cycles, ddCyclStep);
        }

        /* increase the MD step number */
        step++;
        step_rel++;

        resetHandler->resetCounters(
                step, step_rel, mdlog, fplog, cr, fr->nbv.get(),
                nrnb, fr->pmedata, pme_loadbal, wcycle, walltime_accounting);

    }
    /* End of main MD loop */

    /* Closing TNG files can include compressing data. Therefore it is good to do that
     * before stopping the time measurements. */
    mdoutf_tng_close(outf);

    /* Stop measuring walltime */
    walltime_accounting_end_time(walltime_accounting);

    if (!thisRankHasDuty(cr, DUTY_PME))
    {
        /* Tell the PME only node to finish */
        gmx_pme_send_finish(cr);
    }

    if (MASTER(cr))
    {
        if (ir->nstcalcenergy > 0)
        {
            energyOutput.printAverages(fplog, groups);
        }
    }
    done_mdoutf(outf);

    if (bPMETune)
    {
        pme_loadbal_done(pme_loadbal, fplog, mdlog, fr->nbv->useGpu());
    }

    done_shellfc(fplog, shellfc, step_rel);

    walltime_accounting_set_nsteps_done(walltime_accounting, step_rel);

    global_stat_destroy(gstat);

}
