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
#ifndef GMX_MDLIB_ENERGYOUTPUT_H
#define GMX_MDLIB_ENERGYOUTPUT_H

#include <cstdio>
#include <functional>

#include "gromacs/mdrun/integratorinterfaces.h"
#include "gromacs/mdtypes/enerdata.h"

class energyhistory_t;
struct ener_file;
struct gmx_ekindata_t;
struct gmx_enerdata_t;
struct gmx_groups_t;
struct gmx_mtop_t;
struct gmx_output_env_t;
struct t_ebin;
struct t_expanded;
struct t_fcdata;
struct t_grpopts;
struct t_inputrec;
struct t_lambda;
class t_state;

namespace gmx
{
class Awh;
class Constraints;
class MDAtoms;
}

extern const char *egrp_nm[egNR+1];

/* delta_h block type enum: the kinds of energies written out. */
enum
{
    dhbtDH   = 0, /* delta H BAR energy difference*/
    dhbtDHDL = 1, /* dH/dlambda derivative */
    dhbtEN,       /* System energy */
    dhbtPV,       /* pV term */
    dhbtEXPANDED, /* expanded ensemble statistics */
    dhbtNR
};

namespace gmx
{

// TODO remove use of detail namespace when removing t_mdebin in
// favour of an Impl class.
namespace detail
{
struct t_mdebin;
}

/* The functions & data structures here determine the content for outputting
   the .edr file; the file format and actual writing is done with functions
   defined in enxio.h */

class EnergyOutput
{
    public:
        EnergyOutput();
        /*! \brief Initiate MD energy bin
         *
         * This second phase of construction is needed until we have
         * modules that understand how to request output from
         * EnergyOutput.
         *
         * \todo Refactor to separate a function to write the energy
         * file header. Perhaps transform the remainder into a factory
         * function.
         */
        void prepare(ener_file        *fp_ene,
                     const gmx_mtop_t *mtop,
                     const t_inputrec *ir,
                     FILE             *fp_dhdl,
                     bool              isRerun = false);
        ~EnergyOutput();
        /*! \brief Update the averaging structures.
         *
         * Called every step on which the energies are evaluated. */
        void addDataAtEnergyStep(bool                    bDoDHDL,
                                 bool                    bSum,
                                 double                  time,
                                 real                    tmass,
                                 gmx_enerdata_t         *enerd,
                                 t_state                *state,
                                 t_lambda               *fep,
                                 t_expanded             *expand,
                                 matrix                  lastbox,
                                 tensor                  svir,
                                 tensor                  fvir,
                                 tensor                  vir,
                                 tensor                  pres,
                                 gmx_ekindata_t         *ekind,
                                 rvec                    mu_tot,
                                 const gmx::Constraints *constr);
        /*! \brief Updated the averaging structures
         *
         * Called every step on which the energies are not evaluated.
         *
         * \todo This schedule is known in advance, and should be made
         * an intrinsic behaviour of EnergyOutput, rather than being
         * wastefully called every step. */
        void recordNonEnergyStep();

        /*! \brief Help write quantites to the energy file
         *
         * \todo Perhaps this responsibility should involve some other
         * object visiting all the contributing objects. */
        void printStepToEnergyFile(ener_file *fp_ene, bool bEne, bool bDR, bool bOR,
                                   FILE *log,
                                   int64_t step, double time,
                                   int mode,
                                   t_fcdata *fcd,
                                   gmx_groups_t *groups, t_grpopts *opts,
                                   gmx::Awh *awh);
        /*! \brief Get the number of energy terms recorded.
         *
         * \todo Refactor this to return the expected output size,
         * rather than exposing the implementation details about
         * energy terms. */
        int numEnergyTerms() const;
        /*! \brief Getter used for testing t_ebin
         *
         * \todo Find a better approach for this. */
        t_ebin *getEbin();

        /* Between .edr writes, the averages are history dependent,
           and that history needs to be retained in checkpoints.
           These functions set/read the energyhistory_t class
           that is written to checkpoints in checkpoint.c */

        //! Fill the energyhistory_t data.
        void fillEnergyHistory(energyhistory_t * enerhist) const;
        //! Restore from energyhistory_t data.
        void restoreFromEnergyHistory(const energyhistory_t &enerhist);

    private:
        // TODO transform this into an impl class.
        detail::t_mdebin *mdebin = nullptr;
};

} // namespace gmx

//! Open the dhdl file for output
FILE *open_dhdl(const char *filename, const t_inputrec *ir,
                const gmx_output_env_t *oenv);

namespace gmx
{

//! Print an energy-output header to the log file
void print_ebin_header(FILE *log, int64_t steps, double time);

/*! \internal
 * \brief Element signalling energy related special steps
 *
 * This element monitors the current step, and informs its clients via callbacks
 * of the following events:
 *   - energy calculation step
 *   - virial calculation step
 *   - energy writing step
 *   - free energy calculation step
 */
class EnergySignaller : public IIntegratorElement, public ILoggingSignallerClient, public ILastStepClient
{
    public:
        //! Constructor
        EnergySignaller(
            StepAccessorPtr stepAccessor,
            int             nstcalcenergy,
            int             nstenergy);

        //! IIntegratorElement functions
        ElementFunctionTypePtr registerSetup() override;
        ElementFunctionTypePtr registerRun() override;
        ElementFunctionTypePtr registerTeardown() override;

        //! Allows clients to register callback for the different events
        void registerCallback(
            EnergySignallerCallbackPtr calculateEnergyCallback,
            EnergySignallerCallbackPtr calculateVirialCallback,
            EnergySignallerCallbackPtr writeEnergyCallback,
            EnergySignallerCallbackPtr calculateFreeEnergyCallback);

        //! Register callback to get informed about last step
        LastStepCallbackPtr getLastStepCallback() override;

        //! Register callback to get informed about logging step
        LoggingSignallerCallbackPtr getLoggingCallback() override;

    private:
        StepAccessorPtr stepAccessor_;
        bool            isLastStep_;
        bool            isLoggingStep_;

        int             nstcalcenergy_;
        int             nstenergy_;

        std::vector<EnergySignallerCallbackPtr> calculateEnergyCallbacks_;
        std::vector<EnergySignallerCallbackPtr> calculateVirialCallbacks_;
        std::vector<EnergySignallerCallbackPtr> writeEnergyCallbacks_;
        std::vector<EnergySignallerCallbackPtr> calculateFreeEnergyCallbacks_;

        /*! Queries the current step via the step accessor, and informs its clients
         * if this is a special step.
         */
        void run();
};

class EnergyElement :
    public IIntegratorElement, public ITrajectoryWriterClient,
    public IEnergySignallerClient, public ILoggingSignallerClient,
    public ILastStepClient
{
    public:
        EnergyElement(
            StepAccessorPtr stepAccessor,
            TimeAccessorPtr timeAccessor,
            gmx_mtop_t     *mtop,
            t_inputrec     *ir,
            MDAtoms        *mdAtoms,
            t_state        *localState,
            gmx_enerdata_t *enerd,
            tensor          force_vir,
            tensor          shake_vir,
            tensor          total_vir,
            tensor          pres,
            gmx_ekindata_t *ekind,
            Constraints    *constr,
            rvec            mu_tot,
            FILE           *fplog,
            t_fcdata       *fcd,
            bool            isMaster);

        // IIntegratorElement
        ElementFunctionTypePtr registerSetup() override;
        ElementFunctionTypePtr registerRun() override;
        ElementFunctionTypePtr registerTeardown() override;

        // ITrajectoryWriterClient
        TrajectoryWriterCallbackPtr registerTrajectoryWriterSetup() override;
        TrajectoryWriterCallbackPtr registerTrajectoryRun() override;
        TrajectoryWriterCallbackPtr registerEnergyRun() override;
        TrajectoryWriterCallbackPtr registerTrajectoryWriterTeardown() override;

        //IEnergySignallerClient
        EnergySignallerCallbackPtr getCalculateEnergyCallback() override;
        EnergySignallerCallbackPtr getCalculateVirialCallback() override;
        EnergySignallerCallbackPtr getWriteEnergyCallback() override;
        EnergySignallerCallbackPtr getCalculateFreeEnergyCallback() override;

        LastStepCallbackPtr getLastStepCallback() override;

        LoggingSignallerCallbackPtr getLoggingCallback() override;

    private:
        EnergyOutput    energyOutput_;

        const bool      isMaster_;
        bool            isEnergyCalculationStep_;
        bool            writeEnergy_;
        bool            isFreeEnergyCalculationStep_;
        bool            writeLog_;
        bool            isLastStep_;

        StepAccessorPtr stepAccessor_;
        TimeAccessorPtr timeAccessor_;

        void step();
        void setup(gmx_mdoutf *outf);
        void write(gmx_mdoutf *outf, bool isTeardown = false);

        //! Contains user input mdp options.
        t_inputrec       *inputrec_;
        //! Full system topology.
        const gmx_mtop_t *top_global_;
        //! Atom parameters for this domain.
        MDAtoms          *mdAtoms_;
        //! The local state
        t_state          *localState_;
        //! Energy data structure
        gmx_enerdata_t   *enerd_;
        //! Virials
        rvec             *force_vir_, *shake_vir_, *total_vir_, *pres_;
        //! The kinetic energy data structure
        gmx_ekindata_t   *ekind_;
        //! Handles constraints.
        Constraints      *constr_;
        //! Total dipole moment (I guess...)
        real            * mu_tot_;
        //! Handles logging.
        FILE             *fplog_;
        //! Helper struct for force calculations.
        t_fcdata         *fcd_;
        //! Global topology groups
        gmx_groups_t     *groups_;
};

} // namespace gmx

#endif
