from slappsim.Petri import *
from slappsim.States import Start
from slappsim.Function import Function, FunctionStart
from slappsim.Structures import Structure
from slappsim.States import End
from typing import List, Dict, Tuple
from functools import partial
import multiprocessing


def zero_delay():
    return 0


class PetriApp(Petri):
    """Serverless application CSPN model

    Conditional stochastic Petri net model of the serverless application

    Attributes:
        functions: A list of serverless functions in the CSPN model (F).
        transitions: A list of transitions in the CSPN model.
        structures: A list of structures in the serverless application (optional).
        delays: A dictionary that contains the partial functions to generate random delay for different delay types.
        pgs: Price per GB-second USD.
        ppi: Price per invocation in USD.

    """

    def __init__(self, functions: List[Function], transitions: List[Transition], structures: List[Structure] = None,
                 delays: Dict[str, partial] = None,
                 pgs: float = 0.0000166667, ppi: float = 0.0000002) -> None:
        """Inits the application CSPN model."""
        self.functions = functions
        self.functions_map = {function.name: function for function in functions if function.name is not None}
        self.transitions = set(transitions)
        self.structures = [transition.parent_structure for transition in self.transitions if
                           transition.parent_structure is not None]
        self.structures = list(set(self.structures))
        Petri.__init__(self, transitions=list(self.transitions))
        self.arcs = [item for transition in self.transitions for item in transition.in_arcs + transition.out_arcs]
        self.places = set([arc.place for arc in self.arcs])
        self.terminals = set([place for place in self.places if issubclass(place.__class__, End)])
        self.cost = None
        self.cost_records = []
        self.pgs = pgs
        self.pmms = self.pgs / 1024 / 1000
        self.ppi = ppi
        self.delays = delays if delays is not None else {}
        self.scheduling_overhead = None
        self.update_delays()

    def update_delays(self) -> None:
        """Updates the delay

        Generates new delays for different types of transitions.
        Initializes the partial function to generate the scheduling overhead of the application.

        """
        zero_delay_fun = partial(zero_delay)
        for transition in self.transitions:
            if transition.label is not None and self.delays.get(transition.label, None) is not None:
                transition.delay_fun = self.delays[transition.label]
                transition.sample()
        self.scheduling_overhead = self.delays.get("SchedulingOverhead", zero_delay_fun)

    def reset(self) -> None:
        """Resets the CSPN model

        Removes all tokens.
        Generates new firing delay (response time) and communication delay for serverless functions.

        """
        for place in self.places:
            if type(place) is Start:
                place.holding = Token(elapsed_time=0)
            else:
                place.holding = None
        for function in self.functions:
            function.sample()

    def execute(self) -> Tuple[float, float, str, List]:
        """Runs the CSPNs.

        Checks the enabled transitions, fires the enabled transitions, and solves the conflicts.

        Returns:
            A 4-tuple that contains the end-to-end response time, total cost, exit status, and firing logs.
        """
        cost_list = []
        terminals_with_holding = []
        firing_logs = []
        while True:
            fired_transitions = self.run()
            for transition in fired_transitions:
                if transition.label == "FunctionExecution" or transition.label == "Task":
                    cost_list.append((transition.last_fired_time, transition.get_cost(self.pmms, self.ppi)))
                    firing_logs.append((transition.uid, transition.last_fired_time, transition.required_time,
                                        transition.get_cost(self.pmms, self.ppi)))
                else:
                    firing_logs.append((transition.uid, transition.last_fired_time, transition.required_time, 0))
            is_terminated = False
            for terminal in self.terminals:
                if terminal.holding is not None:
                    terminals_with_holding.append(terminal)
                    if len(fired_transitions) == 0:
                        is_terminated = True
            if is_terminated:
                break
        terminal_holdings = [terminal.holding for terminal in terminals_with_holding]
        elapsed_time = [token.elapsed_time for token in terminal_holdings]
        terminal_time = min(elapsed_time)
        cost = sum([cost for time, cost in cost_list if time <= terminal_time])
        exit_point = terminals_with_holding[elapsed_time.index(terminal_time)]
        firing_logs = [log for log in firing_logs if log[1] <= terminal_time]
        terminal_time = terminal_time + self.scheduling_overhead()
        return terminal_time, cost, exit_point.label, firing_logs

    def _execute(self, k: int) -> Tuple[List, List, List, List]:
        """Runs the CSPNs k times

        Returns:
            A 4-tuple that contains the list of the end-to-end response time, list of the total cost, list of the exit
            status, and list of the firing logs. Each list has k elements.
        """
        terminal_time = []
        cost = []
        exit_status = []
        firing_logs = []
        for i in range(k):
            self.reset()
            ert, c, status, log = self.execute()
            terminal_time.append(ert)
            cost.append(c)
            exit_status.append(status)
            firing_logs.append(log)
        return terminal_time, cost, exit_status, firing_logs

    def profile(self, k: int, cpu_count: int = None) -> Tuple[List, List, List, List]:
        """Runs the CSPNs k times (parallel computing)"""
        cpu_count = multiprocessing.cpu_count() if cpu_count is None else cpu_count
        pool = multiprocessing.Pool(cpu_count)
        results = pool.map(self._execute, [int(k / cpu_count) for i in range(cpu_count)])
        pool.close()
        pool.join()
        terminal_time = []
        cost = []
        exit_status = []
        firing_logs = []
        for res in results:
            ert, c, status, log = res
            terminal_time += ert
            cost += c
            exit_status += status
            firing_logs += log
        return terminal_time, cost, exit_status, firing_logs
