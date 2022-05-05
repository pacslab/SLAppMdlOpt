from slappsim.Petri import Transition, InArc, OutArc, Place
from slappsim.States import State
from slappsim.Function import Function
from typing import Union, List
import uuid


class Structure:
    """Structure

    Structures in serverless applications that are modelled with CSPNs.

    Attributes:
        label: A string of the structure type.
        parent_structure: An object of the parent structure of the structure.
        structure_start: The start place of the structure.
        structure_end: The end place of the structure.
        transitions: A list of transitions in the structure.
        uid: A string of the structure UID.
        start_transition: The transition that requires tokens from the start place
        end_transition: The transition that gives tokens to the end place of the structure.
        copied_from: A Structure instance from which the structure is copied.

    """

    def __init__(self, label: str = None, parent_structure=None):
        self.structure_start = StructureStart()
        self.structure_end = StructureEnd()
        self.transitions = []
        self.parent_structure = parent_structure
        self.uid = uuid.uuid4().hex
        self.label = label
        self.start_transition = None
        self.end_transition = None
        self.copied_from = None

    def __str__(self):
        return self.uid

    def __repr__(self):
        return self.uid

    def start_place(self) -> Place:
        """Gets the start place of the structure CSPN model."""
        return self.structure_start

    def end_place(self) -> Place:
        """Gets the end place of the structure CSPN model."""
        return self.structure_end


class StructureStart(Place):

    def __init__(self):
        Place.__init__(self, holding=None)
        self.type = 'StructureStart'


class StructureEnd(Place):

    def __init__(self):
        Place.__init__(self, holding=None)
        self.type = 'StructureEnd'


class SequenceStart(Place):

    def __init__(self):
        Place.__init__(self, holding=None)
        self.type = 'SequenceStart'


class SequenceEnd(Place):

    def __init__(self):
        Place.__init__(self, holding=None)
        self.type = 'SequenceEnd'


class ChoiceStart(Place):

    def __init__(self):
        Place.__init__(self, holding=None)
        self.type = 'ChoiceStart'


class ChoiceEnd(Place):

    def __init__(self):
        Place.__init__(self, holding=None)
        self.type = 'ChoiceEnd'


class ParallelStart(Place):

    def __init__(self):
        Place.__init__(self, holding=None)
        self.type = 'ParallelStart'


class ParallelEnd(Place):
    holding = None

    def __init__(self):
        Place.__init__(self, holding=None)
        self.type = 'ParallelEnd'


class Sequence(Structure):
    """Sequence

    Sequences in serverless applications.

    Attributes:
        actions: A list of actions (tasks) in the sequence.

    """

    def __init__(self, actions: List[Union[State, Function, Structure]]):
        """
        Construct a sequence structure.

        Args:
            actions: A list of actions (tasks) in the sequence.

        Returns:
          None
        """
        self.structure_start = SequenceStart()
        self.structure_end = SequenceEnd()
        self.actions = []
        for action in actions:
            action.parent_structure = self
            self.actions = [action for action in actions]
        Structure.__init__(self, label='Sequence')
        in_arc = InArc(place=self.structure_start)
        out_arc = OutArc(place=self.actions[0].start_place())
        transition = Transition(in_arcs=[in_arc], out_arcs=[out_arc], probability=None, required_time=0,
                                label='InvokeSequence', parent_structure=self)
        self.start_transition = transition
        self.transitions.append(transition)
        in_arc = InArc(place=self.actions[-1].end_place())
        out_arc = OutArc(place=self.structure_end)
        transition = Transition(in_arcs=[in_arc], out_arcs=[out_arc], probability=None, required_time=0,
                                label='SequenceCompleted', parent_structure=self)
        self.end_transition = transition
        self.transitions.append(transition)
        for i in range(len(self.actions) - 1):
            in_arc = InArc(place=self.actions[i].end_place())
            out_arc = OutArc(place=self.actions[i + 1].start_place())
            transition = Transition(in_arcs=[in_arc], out_arcs=[out_arc], probability=None, required_time=0,
                                    label='StateTransition', parent_structure=self)
            self.transitions.append(transition)
        for action in self.actions:
            if type(action) is not State:
                self.transitions += action.transitions

    def copy(self):
        """
        Copy a sequence structure.

        Returns:
          A copy of the sequence structure.
        """
        actions = [action.copy() for action in self.actions]
        sequence = Sequence(actions)
        sequence.copied_from = self
        return sequence


class Choice(Structure):
    """Choice

    Choices in serverless applications.

    Attributes:
        choices: A list of branches.
        probabilities: A list of floating numbers of the probability of executing a branch.

    """

    def __init__(self, choices: List[Union[State, Function, Structure]], probabilities: list[float], end: bool = True):
        """
        Construct a choice structure.

        Args:
            choices: A list of branches.
            probabilities: A list of floating numbers of the probability of executing a branch, which should add up to 1.
            end: A boolean object that indicates if the choice has an end place.

        Returns:
          None
        """
        assert round(sum(probabilities), 4) == 1
        assert len(choices) == len(probabilities)
        self.structure_start = ChoiceStart()
        self.structure_end = ChoiceEnd()
        branches_start = Place(holding=None)
        Structure.__init__(self, label='Choice')
        self.choices = choices
        self.probabilities = probabilities
        in_arc = InArc(place=self.structure_start)
        out_arc = OutArc(place=branches_start)
        transition = Transition(in_arcs=[in_arc], out_arcs=[out_arc], probability=None, required_time=0,
                                label='StateTransition', parent_structure=self)
        self.transitions.append(transition)
        self.start_transition = transition
        branches_end = Place(holding=None)
        if end:
            in_arc = InArc(place=branches_end)
            out_arc = OutArc(place=self.structure_end)
            transition = Transition(in_arcs=[in_arc], out_arcs=[out_arc], probability=None, required_time=0,
                                    label='StateTransition', parent_structure=self)
            self.transitions.append(transition)
            self.end_transition = transition
        for choice, probability in zip(choices, probabilities):
            in_arc = InArc(place=branches_start)
            out_arc = OutArc(place=choice.start_place())
            transition = Transition(in_arcs=[in_arc], out_arcs=[out_arc], probability=probability, required_time=0,
                                    label='InvokeChoice', parent_structure=self)
            self.transitions.append(transition)
            if end:
                choice.parent_structure = self
                in_arc = InArc(place=choice.end_place())
                out_arc = OutArc(place=branches_end)
                transition = Transition(in_arcs=[in_arc], out_arcs=[out_arc], probability=None, required_time=0,
                                        label='ChoiceCompleted', parent_structure=self)
                self.transitions.append(transition)
            if not isinstance(choice, State):
                self.transitions += choice.transitions

    def copy(self):
        """
        Copy a choice structure.

        Returns:
          A copy of the choice structure.
        """
        choices = [choice.copy() for choice in self.choices]
        choice = Choice(choices, probabilities=self.probabilities)
        choice.copied_from = self
        return choice


class Parallel(Structure):
    """Parallel

    Parallels in serverless applications.

    Attributes:
        branches: A list of branches.

    """

    def __init__(self, branches: List[Sequence]):
        """
        Construct a parallel structure.

        Args:
          branches: A list of branches.

        Returns:
          None
        """
        self.structure_start = ParallelStart()
        self.structure_end = ParallelEnd()
        Structure.__init__(self, label='Parallel')
        self.branches = [branch for branch in branches]
        in_arc_start = InArc(place=self.structure_start)
        out_arcs_start = []
        out_arc_end = OutArc(place=self.structure_end)
        in_arcs_end = []
        for branch in self.branches:
            branch.parent_structure = self
            in_arc = InArc(place=branch.structure_end)
            in_arcs_end.append(in_arc)
            out_arc = OutArc(place=branch.structure_start)
            out_arcs_start.append(out_arc)
            self.transitions += branch.transitions
        transition_start = Transition(in_arcs=[in_arc_start], out_arcs=out_arcs_start, probability=None,
                                      required_time=0, label='InvokeParallel', parent_structure=self)
        self.start_transition = transition_start
        transition_end = Transition(in_arcs=in_arcs_end, out_arcs=[out_arc_end], label='ParallelCompleted',
                                    parent_structure=self)
        self.end_transition = transition_end
        self.transitions += [transition_start, transition_end]

    def copy(self):
        """
        Copy a parallel structure.

        Returns:
          A copy of the parallel structure.
        """
        branches = [branch.copy() for branch in self.branches]
        parallel = Parallel(branches)
        parallel.copied_from = self
        return parallel


class Map(Structure):
    """Map

    Maps in serverless applications.

    Attributes:
        sequence: A sequence to iterate.
        iterations: An integer of the number of iterations to run.
        maximum_concurrency: An integer of the upper bound for how many iterations may run in parallel. 0 means no limit.

    """

    def __init__(self, sequence: Sequence, iterations: int, maximum_concurrency: int = 0):
        """
        Construct a map structure.

        Args:
          sequence: A sequence to iterate.
          iterations: An integer of the number of iterations to run.
          maximum_concurrency: An integer of the upper bound for how many iterations may run in parallel. 0 means no limit.

        Returns:
          None
        """
        Structure.__init__(self, label='Map')
        self.sequence = sequence
        self.iterations = iterations
        self.maximum_concurrency = maximum_concurrency
        in_arc_start = InArc(place=self.structure_start)
        out_arc_end = OutArc(place=self.structure_end)
        if maximum_concurrency == 1:
            structures_to_concatenate = [sequence for i in range(iterations)]
        elif iterations <= maximum_concurrency or maximum_concurrency == 0:
            branches = [sequence.copy() for i in range(iterations)]
            parallel = Parallel(branches=branches)
            structures_to_concatenate = [parallel]
            concatenated_structures = Sequence(actions=structures_to_concatenate)
            concatenated_structures.parent_structure = self
            self.transitions += concatenated_structures.transitions
            out_arc_start = OutArc(place=concatenated_structures.structure_start)
            in_arc_end = InArc(place=concatenated_structures.structure_end)
            transition_start = Transition(in_arcs=[in_arc_start], out_arcs=[out_arc_start], label='InvokeMap',
                                          parent_structure=self)
            transition_end = Transition(in_arcs=[in_arc_end], out_arcs=[out_arc_end], label='MapCompleted',
                                        parent_structure=self)
            self.transitions += [transition_start, transition_end]
            self.start_transition = transition_start
            self.end_transition = transition_end
            return
        else:
            structures_to_concatenate = []
            num_parallel = int(iterations / maximum_concurrency)
            for i in range(num_parallel):
                branches = [sequence.copy() for k in range(maximum_concurrency)]
                parallel = Parallel(branches=branches)
                structures_to_concatenate.append(parallel)
            branches = [sequence.copy() for k in range(iterations % maximum_concurrency)]
            parallel = Parallel(branches=branches)
            structures_to_concatenate.append(parallel)
        concatenated_structures = Sequence(actions=structures_to_concatenate)
        concatenated_structures.parent_structure = self
        self.transitions += concatenated_structures.transitions
        out_arc_start = OutArc(place=concatenated_structures.structure_start)
        in_arc_end = InArc(place=concatenated_structures.structure_end)
        transition_start = Transition(in_arcs=[in_arc_start], out_arcs=[out_arc_start], label='InvokeMap',
                                      parent_structure=self)
        transition_end = Transition(in_arcs=[in_arc_end], out_arcs=[out_arc_end], label='MapCompleted',
                                    parent_structure=self)
        self.transitions += [transition_start, transition_end]
        self.start_transition = transition_start
        self.end_transition = transition_end
        pass

    def copy(self):
        """
        Copy a map structure.

        Returns:
          A copy of the map structure.
        """
        map_copy = Map(self.sequence, self.iterations, self.maximum_concurrency)
        map_copy.copied_from = self
        return map_copy
