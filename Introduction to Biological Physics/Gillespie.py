import numpy as np

class StochSimulation:
    """
    implement Gillespie algorithm for stochastic simulation, assuming mass action kinetics.
    available methods include 'direct', 'first-reaction', and 'tau-leaping'.
    based on (D. T. Gillespie, Annu. Rev. Phys. Chem. 58, 2007).
    """

    def __init__(self, stoich, rates, init, record=True):
        """
        general purpose code for stochastic simulation using Gillespie algorithm.
        inputs:
        stoich: 2-tuple, stoichiometry matrices of reactants and products, (R_ui, P_ui), u = 1 ~ m, i = 1 ~ n
        rates: list, rate parameters, K_u, u = 1 ~ M
        init: list, initial numbers of every species, N_i(0), i = 1 ~ n
        record: boolean, whether to record full history of events
        """
        self.reactants = np.asarray(stoich[0])    # stoichiometry matrices of reactants
        self.products = np.asarray(stoich[1])    # stoichiometry matrices of products
        self.rates = np.asarray(rates, dtype=float)    # rate constant for each reaction
        self.numbers = np.asarray(init, dtype=int)    # current number of each species

        self.num_reac = self.reactants.shape[0]    # number of reactions
        self.num_spec = self.reactants.shape[1]    # number of species

        self.time = 0.                  # time since beginning of simulation
        self.nevents = 0                # total number of events that have happened
        self.term = False               # whether reactions terminated
        self.record = record            # whether to record time series

        if self.record:
            self.time_hist = [0]                  # list of times at which events happened
            self.event_hist = [-1]                # list of events that happened, -1 represents initial time
            self.numbers_hist = [self.numbers.copy()]     # list of species_number arrays right after each event


    def run(self, tmax, maxstep=10000, nmax=1000, disp=0, method='first-reaction'):
        """
        run simulation until time `tmax` since the beginning of the simulation.
        inputs:
        tmax: float, time since the beginning of the simulation.
        maxstep: int, maximum number of steps to simulate even if `tmax` is not reached.
        nmax: int, maximum number of agents of any species at which simulation stops.
        disp: int, print messages if >= 0, higher values allow more details.
        method: 'direct'|'first-reaction'|'tau-leaping', method to use.
        """
        self.term = False
        for n in range(maxstep):                # bound on number of steps
            if self.time >= tmax:               # bound on accumulated time
                return
            if np.any(self.numbers >= nmax):    # bound on number of each species
                if disp >= 0:
                    print('maximum number of agents reached.')
                return
            a_j = self.rates * np.prod(np.power(self.numbers, self.reactants), axis=1)    # mass action
            if np.all(a_j == 0):
                self.term = True    # reaction terminated
                if disp >= 0:
                    print('reactions terminated.')
                self.time = tmax    # jump to final time
                return
            elif np.any(a_j < 0):    # should not happen
                raise RuntimeError('transition rates become negative!')
            a_j = np.maximum(a_j, 1e-15)    # if a_j is zero (when N=0), replace by a small number to avoid division by 0
            if method == 'direct':  # direct method
                dn_i, tau, events = self.direct(a_j)
            elif method == 'first-reaction':  # first-reaction method
                dn_i, tau, events = self.first_react(a_j)
            elif method == 'tau-leaping':  # tau-leaping method, requires parameter tau
                tau = keywords['tau']
                dn_i, tau, events = self.tau_leap(a_j, tau)
            self.numbers += dn_i
            self.time += tau
            self.nevents += len(events)
            if self.record:
                self.time_hist.extend([self.time for ev in events])
                self.event_hist.extend(events)
                self.numbers_hist.extend([self.numbers.copy() for ev in events])
            if disp > 0:
                for i in range(len(events)):
                    ev = self.nevents - len(events) + i + 1
                    print(f'event {ev}: t = {self.time}, triggering reaction {events[i]}')
                    if disp > 1:
                        print(f'current numbers = {self.numbers}')
        else:
            if disp >= 0:
                print('maximum number of steps reached.')


    def direct(self, a_j):  # direct method
        a0 = np.sum(a_j)
        tau = -1. / a0 * np.log(np.random.rand())
        prob = a_j / a0
        event = np.random.choice(self.num_reac, p=prob)
        dn_i = self.products[event] - self.reactants[event]
        return dn_i, tau, [event]


    def first_react(self, a_j):     # first-reaction method
        tau_j = -1. / a_j * np.log(np.random.rand(self.num_reac))
        event = np.argmin(tau_j)
        dn_i = self.products[event] - self.reactants[event]
        tau = tau_j[event]
        return dn_i, tau, [event]


    def tau_leap(self, a_j, tau):   # tau-leaping method
        r_j = np.random.poisson(a_j*tau)
        dn_i = np.dot(self.products - self.reactants, r_j)
        if np.any(np.abs(dn_i) > np.maximum(1, 0.03*self.numbers)):    # check leap condition
            print('Warning: leap condition violated.')
        if np.any(self.numbers + dn_i < 0):
            raise RuntimeError('numbers become negative during tau-leaping.')
        events = np.repeat(np.arange(self.num_reac), r_j)
        return dn_i, tau, events
