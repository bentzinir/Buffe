#!/usr/bin/python

import struct
import os

from collections import namedtuple

NUMBER_OF_TARGETS = 5
#NUMBER_OF_STATES = NUMBER_OF_TARGETS*3 + 3
NUMBER_OF_STATES = 4
STATE_MSG_FORMAT = 'f'*NUMBER_OF_STATES
MSG_SIZE = struct.calcsize(STATE_MSG_FORMAT)

class SIMULATOR(object):
    def __init__(self):
        pass

    def _open_pipes(self):
        # return None

        PipesTuple = namedtuple('pipe_t', ['sim_to_tf_p', 'tf_to_sim_p'])
        pipes = PipesTuple(sim_to_tf_p=os.open("./sim_to_tf", os.O_WRONLY),
                           tf_to_sim_p=os.open("./tf_to_sim", os.O_RDONLY))
        return pipes

    def run(self):
        pipes = self._open_pipes()

        while True:
            # get message
            statebuf = os.read(pipes.tf_to_sim_p, MSG_SIZE)
            income_state = struct.unpack(STATE_MSG_FORMAT, statebuf)

            # get policy actions
            outcome_state = income_state[::-1]

            # send state
            buf = struct.pack('ffff', *outcome_state)
            os.write(pipes.sim_to_tf_p, buf)
