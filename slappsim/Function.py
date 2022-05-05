import math
import uuid
from functools import partial

from slappsim.Petri import *


class FunctionStart(Place):
    def __init__(self):
        Place.__init__(self, holding=None)
        self.label = 'FunctionStart'


class FunctionEnd(Place):
    def __init__(self):
        Place.__init__(self, holding=None)
        self.label = 'FunctionEnd'


class Function(Transition):
    """Serverless function

    CSPN model for Serverless functions and actions.

    Attributes:
        mem: An integer of the size of the allocated memory size in megabytes.
        rt: A floating number of the response time in milliseconds.
        name: A string of the name of the serverless function.
        pf_fun: A partial function that generates the random response time based on the performance profile.
        delay_fun: A partial function that generates the random communication delay.
        copied_from: A Function instance from which the function is copied.
    """

    def __init__(self, mem: int = 128, pf_fun: partial = None, rt: float = None, name: str = None,
                 copied_from=None, delay_fun: partial = None):
        self.mem = mem
        self.name = name
        self.copied_from = copied_from
        if self.name is None:
            self.name = uuid.uuid4().hex
        assert not (pf_fun is None and rt is None)
        if pf_fun is None:
            self.rt = rt
            self.pf_fun = None
        else:
            self.pf_fun = pf_fun
            self.rt = self.pf_fun()
        self.function_start = FunctionStart()
        self.function_end = FunctionEnd()
        self.in_arc = InArc(place=self.function_start, amount=1)
        self.out_arc = OutArc(place=self.function_end, amount=1)
        Transition.__init__(self, in_arcs=[self.in_arc], out_arcs=[self.out_arc], required_time=self.rt,
                            label='FunctionExecution', delay_fun=delay_fun)
        self.transitions = [self]
        self.delay_fun = delay_fun
        # if delay_fun is not passed, the default communication delay is 0 milliseconds.
        self.delay = 0
        self.cost = 0

    def start_place(self) -> Place:
        """Gets the start place of the function CSPN model."""
        return self.function_start

    def end_place(self) -> Place:
        """Gets the end place of the function CSPN model."""
        return self.function_end

    def sample(self) -> float:
        """Samples the firing delay (response time)."""
        Transition.sample(self)
        if self.pf_fun is not None:
            self.rt = self.pf_fun()
            self.required_time = self.rt
        return self.rt

    def get_cost(self, pmms: float, ppi: float) -> float:
        """Calculates the cost in USD."""
        #  AWS Lambda Pricing Model: https://aws.amazon.com/lambda/pricing/
        self.cost = math.ceil(self.rt) * self.mem * pmms + ppi
        return self.cost

    def calculate_cost(self, rt: float, mem: float, pmms: float, ppi: float) -> float:
        """Calculates the cost in USD (static method)."""
        return math.ceil(rt) * mem * pmms + ppi

    def copy(self):
        """Creates a copy of the serverless function."""
        function = Function(pf_fun=self.pf_fun, rt=self.rt, mem=self.mem, name=self.name, delay_fun=self.delay_fun)
        function.copied_from = self
        return function

    def fire(self):
        """Simulates the execution of the serverless function in CSPNs."""
        self.sample()
        Transition.fire(self)


class Pass(Function):
    """Pass state

    CSPN model for pass states.

    """

    def __init__(self):
        Function.__init__(self, mem=0, rt=0)
        self.label = 'Pass'

    def get_cost(self, *args, **kwargs) -> float:
        return 0


class Wait(Function):
    """Wait state

    CSPN model for wait states.

    """

    def __init__(self, rt: float):
        Function.__init__(self, mem=0, rt=rt)
        self.label = 'Wait'

    def get_cost(self, *args, **kwargs) -> float:
        return 0


class Task(Function):
    """Task

    CSPN model for actions with different billing models.

    """

    def __init__(self, rt: float, cost: float = 0):
        Function.__init__(self, rt=rt)
        self.cost = cost
        self.label = 'Task'

    def get_cost(self, *args, **kwargs) -> float:
        return self.cost
