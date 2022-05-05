from typing import List
from collections import defaultdict
import random
from functools import partial
import uuid


class Token:
    """Token

    Tokens in CSPNs.

    Attributes:
        elapsed_time: a floating number of the amount of elapsed time on the token in milliseconds (U).

    """

    def __init__(self, elapsed_time: float = 0):
        self.elapsed_time = elapsed_time


class Place:
    """Place

    Places in CSPNs.

    Attributes:
        holding: An integer of the size of the allocated memory size in megabytes.
        label: A floating number of the response time in milliseconds.
        parent_structure: A string of the name of the serverless function.
    """

    def __init__(self, holding: Token = None, label: str = None, parent_structure=None):
        self.holding = holding
        self.label = label
        self.parent_structure = parent_structure
        assert label in [None, 'ChoiceStart', 'ChoiceEnd',
                         'ParallelStart', 'ParallelEnd', 'MapStart', 'MapEnd',
                         'FunctionStart', 'FunctionEnd', 'TaskStart', 'TaskEnd',
                         'PassStart', 'PassEnd', 'WaitStart', 'WaitEnd', 'SequenceStart',
                         'SequenceEnd', 'Start', 'End', 'Succeed', 'Fail']


class Arc:
    """Arc

    Arcs in CSPNs.

    Attributes:
        place: A place from/to which the transition takes/gives tokens.
        amount: An integer of the number of tokens that are consumed/produced.
    """

    def __init__(self, place: Place, amount: int = 1):
        self.place = place
        self.amount = amount


class InArc(Arc):
    """InArc

    Input arcs in CSPNs.

    """

    def trigger(self) -> Token:
        """Consume the token."""
        token_to_fire = self.place.holding
        self.place.holding = None
        return token_to_fire


class OutArc(Arc):
    """OutArc

    Output arcs in CSPNs.

    """

    def trigger(self, token: Token) -> None:
        """Place the token."""
        self.place.holding = token


class Transition:
    """Transition (T)

    Transitions in CSPNs.

    Attributes:
        in_arcs: A list of input arcs (I).
        out_arcs: A list of output arcs (O).
        probability: A floating number of the probability of firing the transition under conflict (Pr).
        required_time: A floating number of the firing delay in milliseconds (W).
        delay: A floating number of the delay added to the firing delay in milliseconds.
        delay_fun: A partial function that generates the random (scheduling) delay.
        label: A string of the transition type.
        parent_structure: An object of the parent structure of the transition.
    """

    def __init__(self, in_arcs: list[InArc], out_arcs: list[OutArc], probability: float = None,
                 required_time: float = 0, delay: float = 0, delay_fun: partial = None,
                 label: str = None, parent_structure=None) -> None:
        self.in_arcs = in_arcs
        self.out_arcs = out_arcs
        self.probability = probability
        self.required_time = required_time
        self.label = label
        self.uid = uuid.uuid4().hex
        self.parent_structure = parent_structure
        self.last_fired_time = 0
        assert not (delay_fun is None and delay is None)
        if delay_fun is None:
            self.delay = delay
            self.delay_fun = None
        else:
            self.delay_fun = delay_fun
            self.delay = self.delay_fun()

        assert label in [None, 'FunctionExecution', 'InvokeChoice', 'ChoiceCompleted',
                         'InvokeParallel', 'ParallelCompleted', 'InvokeMap', 'MapCompleted', 'StateTransition',
                         'Function', 'Pass', 'Wait', 'Task', 'InvokeSequence',
                         'SequenceCompleted']

    def enabled(self) -> bool:
        """Checks if the transition is enabled."""
        return all([in_arc.place.holding for in_arc in self.in_arcs])

    def fire(self) -> None:
        """Fires the transition."""
        max_elapsed_time = 0
        for in_arc in self.in_arcs:
            token = in_arc.trigger()
            max_elapsed_time = token.elapsed_time if token.elapsed_time > max_elapsed_time else max_elapsed_time
        elapsed_time = max_elapsed_time + self.required_time + self.delay
        self.last_fired_time = max_elapsed_time
        for out_arc in self.out_arcs:
            token = Token(elapsed_time=elapsed_time)
            out_arc.trigger(token)

    def sample(self) -> float:
        """Samples the delay."""
        if self.delay_fun is not None:
            self.delay = self.delay_fun()
        return self.delay

    def get_cost(self, *args, **kwargs) -> float:
        """Calculates the cost."""
        return 0

    def __str__(self):
        return self.uid

    def __repr__(self):
        return self.uid


class Petri:
    """CSPNs

    Conditional Stochastic Petri Nets.

    Attributes:
        transitions: A list of transitions in the CSPNs.
    """

    def __init__(self, transitions: list[Transition]) -> None:
        self.transitions = transitions

    def run(self) -> List[Transition]:
        """Runs the CSPNs (one step).

        Checks the enabled transitions, fires the enabled transitions, and solves the conflicts.

        Returns:
            A list of the fired transitions.
        """
        fired_transitions = []
        enabled_transitions = [transition for transition in self.transitions if transition.enabled()]
        enabled_transitions_with_1_in_arc = [(transition.in_arcs[0].place, transition) for transition in
                                             enabled_transitions if len(transition.in_arcs) == 1]
        conflicting_transitions = defaultdict(list)
        for place, transition in enabled_transitions_with_1_in_arc:
            conflicting_transitions[place].append((transition, transition.probability))
        for place in conflicting_transitions.keys():
            if len(conflicting_transitions[place]) == 1:
                continue
            transition_to_fire = random.choices([item[0] for item in conflicting_transitions[place]],
                                                weights=[item[1] for item in conflicting_transitions[place]], k=1)[0]
            transition_to_fire.fire()
            fired_transitions.append(transition_to_fire)
            enabled_transitions = [transition for transition in enabled_transitions if
                                   transition not in [item[0] for item in conflicting_transitions[place]]]
        for transition in enabled_transitions:
            transition.fire()
            fired_transitions.append(transition)
        return fired_transitions
