from slappsim.Petri import *


class State(Place):
    """State

    States in serverless applications

    """

    def start_place(self) -> Place:
        return self

    def end_place(self) -> Place:
        return self


class Start(State):
    """Start state"""

    def __init__(self):
        token = Token(elapsed_time=0)
        Place.__init__(self, holding=token, label='Start')


class End(State):
    """End state"""

    def __init__(self):
        Place.__init__(self, holding=None, label='End')


class Succeed(End):
    """Succeed state"""

    def __init__(self):
        Place.__init__(self, holding=None, label='Succeed')


class Fail(End):
    """Fail state"""

    def __init__(self):
        Place.__init__(self, holding=None, label='Fail')
