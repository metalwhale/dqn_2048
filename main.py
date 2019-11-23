# pylint: skip-file

"""
Main
"""

from dqn_2048 import StateBuilder, Action, Environment

state_builder = StateBuilder().set_size(4).set_unit(2)
environment = Environment(state_builder)
reward = 0

while True:
    transition = environment.execute(Action.random())
    reward += transition.reward
    print(transition.action, reward)
    print(transition.state, "\n")
    if transition.state.is_ended():
        break
