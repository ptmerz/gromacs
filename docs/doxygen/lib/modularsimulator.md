The modular simulator {#page_modularsimulator}
==============================================

A new modular approach to the GROMACS simulator is described. The
simulator in GROMACS is the object which carries out a simulation. The
simulator object is created and owned by the runner object, which is
outside of the scope of this new approach, and will hence not be further
described. The simulator object provides access to some generally used
data, most of which is owned by the runner object.

## Legacy implementation

In the legacy implementation, the simulator consisted of a number of
independent functions carrying out different type of simulations, such
as `do_md` (MD simulations), `do_cg` and `do_steep` (minimization),
`do_rerun` (force and energy evaluation of simulation trajectories),
`do_mimic` (MiMiC QM/MM simulations), `do_nm` (normal mode analysis),
and `do_tpi` (test-particle insertion).

The legacy approach has some obvious drawbacks:
* *Data management:* Each of the `do_*` functions defines local data,
  including complex objects encapsulating some data and functionality,
  but also data structures effectively used as "global variables" for
  communication between different parts of the simulation. Neither the
  ownership nor the access rights (except for `const` qualifiers) are
  clearly defined.
* *Dependencies:* Many function calls in the `do_*` functions are
  dependent on each others, i.e. rely on being called in a specific
  order, but these dependencies are not clearly defined.
* *Branches:* The flow of the `do_*` functions are hard to understand
  due to branching. At setup time, and then at every step of the
  simulation run, a number of booleans are set (e.g. `bNS` (do neighbor
  searching), `bCalcEner` (calculate energies), `do_ene` (write
  energies), `bEner` (energy calculation needed), etc). These booleans
  enable or disable branches of the code (for the current step or the
  entire run), mostly encoded as `if(...)` statements in the main `do_*`
  loop, but also in functions called from there.
* *Task scheduling:* Poorly defined dependencies and per-step branching
  make task scheduling (e.g. parallel execution of independent tasks)
  very difficult.
* *Error-prone for developers:* Poorly defined dependencies and unclear
  code flow make changing the simulator functions very error-prone,
  rendering the implementation of new methods tedious.

## The modular simulator approach

The main design goals of the new, fully modular simulator approach
include
* *Extensibility:* We want to ease maintenance and the implementation
  of new integrator schemes.
* *Monte Carlo:* We want to add MC capability, which can be mixed with
  MD to create hybrid MC/MD schemes.
* *Data locality & interfaces:* We aim at localizing data in objects,
  and offer interfaces if access from other objects is needed.
* *Multi-stepping:* We aim at a design which intrinsically supports
  multi-step integrators, e.g. having force calls at different
  frequencies, or avoid having branches including rare events
  (trajectory writing, neighbor search, ...) in the computation loops.
* *Task parallelism:* Although not first priority, we want to have a
  design which can be extended to allow for task parallelism.

The general design approach is that of a **task scheduler**. *Tasks*
are argument-less functions which perform a part of the computation.
Periodically during the simulation, the scheduler builds a
*queue of tasks*, i.e. a list of tasks which is then run through in
order. Over time, with data dependencies clearly defined, this
approach can be modified to have independent tasks run in parallel.

The approach is most easily displayed using some pseudo code:

    class ModularSimulator : public ISimulator
    {
        public:
            //! Run the simulator
            void run() override;
        private:
            std::vector<ISignaller*> signallers_;
            std::vector<ISimulatorElement*> elements_;
            std::queue<SimulatorRunFunction*> taskQueue_;
    }

    void ModularSimulator::run()
    {
        constructElementsAndSignallers();
        setupAllElements();
        while (not lastStep)
        {
            // Fill the task queue with new tasks (can be precomputed for many steps)
            populateTaskQueue();
            // Now simply loop through the queue and run one task after the next
            for (auto task : taskQueue)
            {
                (*task)();  // run task
            }
        }
    }

This allows for an important division of tasks.

* `constructElementsAndSignallers()` is responsible to **store the
  elements in the right order**. This includes the different order of
  element in different algorithms (e.g. leap-frog vs. velocity
  verlet), but also logical dependencies (energy output after compute
  globals).
* `populateTaskQueue()` is responsible to **decide if elements need to
  run at a specific time step**. The elements get called in order, and
  decide whether they need to run at a specific step. This can be
  pre-computed for multiple steps. In the current implementation, the
  tasks are pre-computed for the entire life-time of the neighbor
  list.
* **Running the actual simulation tasks** is done after the task queue
  was filled.  This is achieved by simply looping over the task list,
  no conditionals or branching needed.

### Simulator elements

The task scheduler holds a list of *simulator elements*, defined by
the `ISimulatorElement` interface. These elements have a
`scheduleTask(Step, Time)` function, which gets called by the task
scheduler. This allows the simulator element to register one (or more)
function pointers to be run at that specific `(Step, Time)`. From the
point of view of the element, it is important to note that the
computation will not be carried out immediately, but that it will be
called later during the actual (partial) simulation run. From the
point of view of the builder of the task scheduler, it is important to
note that the order of the elements determines the order in which
computation is performed. The task scheduler periodically loops over
its list of elements, builds a queue of function pointers to run, and
returns this list of tasks. As an example, a possible application
would be to build a new queue after each domain-decomposition (DD) /
neighbor-searching (NS) step, which might occur every 100 steps. The
scheduler would loop repeatedly over all its elements, with elements
like the trajectory-writing element registering for only one or no
step at all, the energy-calculation element registering for every
tenth step, and the force, position / velocity propagation, and
constraining algorithms registering for every step. The result would
be a (long) queue of function pointers including all computations
needed until the next DD / NS step, which can be run without any
branching.

### Signallers

Some elements might require computations by other elements. If for
example, the trajectory writing is an element independent from the
energy-calculation element, it needs to signal to the energy element
that it is about to write a trajectory, and that the energy element
should be ready for that (i.e. perform an energy calculation in the
upcoming step). This requirement, which replaces the boolean branching
in the current implementation, is fulfilled by a Signaller - Client
model. Classes implementing the `ISignaller` interface get called
*before* every loop of the element list, and can inform registered
clients about things happening during that step. The trajectory
element, for example, can tell the energy element that it will write
to trajectory at the end of this step. The energy element can then
register an energy calculation during that step, being ready to write
to trajectory when requested.

### Sequence diagrams

#### Pre-loop
In the loop preparation, the signallers and elements are created and
stored in the right order. The signallers and elements can then
perform any setup operations needed.

\msc
hscale="2";

ModularSimulator,
Signallers [label="ModularSimulator::\nSignallers"],
Elements [label="ModularSimulator::\nElements"],
TaskQueue [label="ModularSimulator::\nTaskQueue"];

--- [ label = "constructElementsAndSignallers()" ];
    ModularSimulator => Signallers [ label = "Create signallers\nand order them" ];
    ModularSimulator => Elements [ label = "Create elements\nand order them" ];
--- [ label = "constructElementsAndSignallers()" ];
|||;
|||;

--- [ label = "setupAllElements()" ];
    ModularSimulator => Signallers [ label = "Call setup()" ];
    Signallers box Signallers [ label = "for signaler in Signallers\n    signaller->setup()" ];
    |||;
    ModularSimulator => Elements [ label = "Call setup()" ];
    Elements box Elements [ label = "for element in Elements\n    element->setup()" ];
--- [ label = "setupAllElements()" ];
\endmsc

#### Main loop
The main loop consists of two parts which are alternately run until the
simulation stop criterion is met. The first part is the population of
the task queue, which determines all tasks that will have to run to
simulate the system for a given time period. In the current implementation,
the scheduling period is set equal to the lifetime of the neighborlist.
Once the tasks have been predetermined, the simulator runs them in order.
This is the actual simulation computation, which can now run without any
branching.

\msc
hscale="2";

ModularSimulator,
Signallers [label="ModularSimulator::\nSignallers"],
Elements [label="ModularSimulator::\nElements"],
TaskQueue [label="ModularSimulator::\nTaskQueue"];

ModularSimulator box TaskQueue [ label = "loop: while(not lastStep)" ];
ModularSimulator note TaskQueue [ label = "The task queue is empty. The simulation state is at step N.", textbgcolor="yellow" ];
|||;
|||;
ModularSimulator box ModularSimulator [ label = "populateTaskQueue()" ];
ModularSimulator =>> TaskQueue [ label = "Fill task queue with tasks until next neighbor-searching step" ];
|||;
|||;
ModularSimulator note TaskQueue [ label = "The task queue now holds all tasks needed to move the simulation from step N to step N + nstlist. The simulation for these steps has not been performed yet, however. The simulation state is hence still at step N.", textbgcolor="yellow" ];
|||;
|||;

ModularSimulator => TaskQueue [ label = "Run all tasks in TaskQueue" ];
TaskQueue box TaskQueue [label = "for task in TaskQueue\n    run task" ];
TaskQueue note TaskQueue [ label = "All simulation computations are happening in this loop!", textbgcolor="yellow" ];
|||;
|||;
ModularSimulator note TaskQueue [ label = "The task queue is now empty. The simulation state is at step N + nstlist.", textbgcolor="yellow" ];
ModularSimulator box TaskQueue [ label = "end loop: while(not lastStep)" ];

\endmsc

#### Task scheduling
A part of the main loop, the task scheduling in `populateTaskQueue()` 
allows the elements to push tasks to the task queue. For every scheduling 
step, the signallers are run first to give the elements information about 
the upcoming scheduling step. The scheduling routine elements are then 
called in order, allowing the elements to register their respective tasks.

\msc
hscale="2";

ModularSimulator,
Signallers [label="ModularSimulator::\nSignallers"],
Elements [label="ModularSimulator::\nElements"],
TaskQueue [label="ModularSimulator::\nTaskQueue"];

--- [ label = "populateTaskQueue()" ];
    ModularSimulator box ModularSimulator [ label = "doDomainDecomposition()\ndoPmeLoadBalancing()" ];
    ModularSimulator =>> Elements [ label = "Update state and topology" ];
    |||;
    |||;

    ModularSimulator note ModularSimulator [ label = "schedulingStep == N\nsimulationStep == N", textbgcolor="yellow" ];
    ModularSimulator box TaskQueue [ label = "loop: while(not nextNeighborSearchingStep)" ];
        ModularSimulator => Signallers [ label = "Run signallers for schedulingStep" ];
        Signallers box Signallers [label = "for signaller in Signallers\n    signaller->signal(scheduleStep)" ];
        Signallers =>> Elements [ label = "notify" ];
        Signallers note Elements [ label = "The elements now know if schedulingStep has anything special happening, e.g. neighbor searching, log writing, trajectory writing, ...", textbgcolor="yellow" ];
        |||;
        |||;

        ModularSimulator => Elements [ label = "Schedule run functions for schedulingStep" ];
        Elements box Elements [label = "for element in Elements\n    element->scheduleTask(scheduleStep)" ];
        Elements =>> TaskQueue [ label = "Push task" ];
        Elements note TaskQueue [ label = "The elements have now registered everything they will need to do for schedulingStep.", textbgcolor="yellow" ];
        ModularSimulator note ModularSimulator [ label = "schedulingStep++", textbgcolor="yellow" ];

    ModularSimulator box TaskQueue [ label = "end loop: while(not nextNeighborSearchingStep)" ];
--- [ label = "populateTaskQueue()" ];
ModularSimulator note ModularSimulator [ label = "schedulingStep == N + nstlist\nsimulationStep == N", textbgcolor="yellow" ];

\endmsc

## Acceptance tests and further plans

Acceptance tests before this can be made default code path (as
defined with Mark Jan 2019)
* End-to-end tests pass on both `do_md` and the new loop in
  Jenkins pre- and post-submit matrices
* Physical validation cases pass on the new loop
* Performance on different sized benchmark cases, x86 CPU-only
  and NVIDIA GPU are at most 1% slower -
  https://github.com/ptmerz/gmxbenchmark has been developed to
  this purpose.

After the NVE MD bare minimum, we will want to add support for
* Thermo- / barostats
* FEP
* Pulling
* Checkpointing

Using the new modular simulator framework, we will then explore
adding new functionality to GROMACS, including
* Monte Carlo barostat
* hybrid MC/MD schemes
* multiple-time-stepping integration

We sill also explore optimization opportunities, including
* re-use of the same queue if conditions created by user input are 
  sufficiently favorable (e.g. by design or when observed)
* simultaneous execution of independent tasks

We will probably not prioritize support for (and might consider
deprecating from do_md for GROMACS 2020)
* Simulated annealing
* REMD
* Simulated tempering
* Multi-sim
* Membrane embedding
* QM/MM
* FEP lambda vectors
* Fancy mdp options for FEP output
* MTTK
* Shell particles
* Essential dynamics
* Enforced rotation
* Constant acceleration groups
* Ensemble-averaged restraints
* Time-averaged restraints
* Freeze, deform, cos-acceleration

## Signallers and elements

The current implementation of the modular simulator consists of
the following signallers and elements:

### Signallers

All signallers have a list of pointers to clients, objects that
implement a respective interface and get notified of events the
signaller is communicating.

* `NeighborSearchSignaller`: Informs its clients whether the
  current step is a neighbor-searching step.
* `LastStepSignaller`: Informs its clients when the current step
  is the last step of the simulation.
* `LoggingSignaller`: Informs its clients whether output to the
  log file is written in the current step.
* `EnergySignaller`: Informs its clients about energy related
  special steps, namely energy calculation steps, virial
  calculation steps, and free energy calculation steps.
* `TrajectoryElement`: Informs its clients if writing to
  trajectory (state [x/v/f] and/or energy) is planned for the
  current step. Note that the `TrajectoryElement` is not a
  pure signaller, but also implements the `ISimulatorElement`
  interface (see section "Simulator Elements" below).

### Simulator Elements

#### `TrajectoryElement`
The `TrajectoryElement` is a special element, as it
is both implementing the `ISimulatorElement` and the `ISignaller`
interfaces. During the signaller phase, it is signalling its
_signaller clients_ that the trajectory will be written at the
end of the current step. During the simulator run phase, it is
calling its _trajectory clients_ (which do not necessarily need
to be identical with the signaller clients), passing them a valid
output pointer and letting them write to trajectory. Unlike the
legacy implementation, the trajectory element itself knows nothing
about the data that is written to file - it is only responsible
to inform clients about trajectory steps, and providing a valid
file pointer to the objects that need to write to trajectory.

#### `MicroState`
The `MicroState` takes part in the simulator run, as it might
have to save a valid state at the right moment during the
integration. Placing the MicroState correctly is for now the
duty of the simulator builder - this might be automated later
if we have enough meta-data of the variables (i.e., if
`MicroState` knows at which time the variables currently are,
and can decide when a valid state (full-time step of all
variables) is reached. The `MicroState` is also a client of
both the trajectory signaller and writer - it will save a
state for later writeout during the simulator step if it
knows that trajectory writing will occur later in the step,
and it knows how to write to file given a file pointer by
the `TrajectoryElement`.

## Data structures

### `MicroState`
The `MicroState` contains a little more than the pure
statistical-physical micro state, namely the positions,
velocities, forces, and box matrix, as well as a backup of
the positions and box of the last time step. While it takes
part in the simulator loop to be able to backup positions /
boxes and save the current state if needed, it's main purpose
is to offer access to its data via getter methods. All elements
reading or writing to this data need a pointer to the
`MicroState` and need to request their data explicitly. This
will later simplify the understanding of data dependencies
between elements.

Note that the `MicroState` can be converted to and from the
legacy `t_state` object. This is useful when dealing with
functionality which has not yet been adapted to use the new
data approach - of the elements currently implemented, only
domain decomposition, PME load balancing, and the initial
constraining are using this.