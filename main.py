# pylint: skip-file

# %%
"""
Main
"""

from sys import stdout
from keras import Model
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from dqn_2048 import Direction, StateBuilder, Environment, QualityBuilder, Agent

BOARD_SIZE = 4
BOARD_UNIT = 2

GAMMA = 0.99
LEARNING_RATE = 1e-4

BATCH_SIZE = EXPERIENCES_COUNT = STARTING_STEP = 500
TARGET_SYNCING_FREQUENCY = 2500
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY_RATE = 50000

STEPS_COUNT = 100000

def model_builder(output_size: int) -> Model:
    """
    # Arguments
        output_size: int. Output size.
    # Returns the model.
    """
    model = Sequential()
    model.add(Dense(1024, activation="relu", input_shape=(BOARD_SIZE ** 2,)))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(output_size))
    model.compile("sgd", loss="mse")
    return model

state_builder = StateBuilder().set_size(BOARD_SIZE).set_unit(BOARD_UNIT)
environment = Environment(state_builder)

quality_builder = QualityBuilder() \
    .set_gamma(GAMMA) \
    .set_output_size(len(Direction)) \
    .set_model_builder(model_builder) \
    .set_optimizer(SGD(lr=1e-4))
agent = Agent(
    quality_builder,
    BATCH_SIZE, EXPERIENCES_COUNT, STARTING_STEP, TARGET_SYNCING_FREQUENCY
)
agent.set_epsilons(EPSILON_START, EPSILON_END, EPSILON_DECAY_RATE)

result = open("result.txt", "w+")
for step in range(STEPS_COUNT):
    agent.observe(environment)
    if step % 100 == 0:
        stdout.write(str(step) + "\n")
        stdout.flush()
    # Play and save weights before syncing
    if step % TARGET_SYNCING_FREQUENCY == TARGET_SYNCING_FREQUENCY - 1:
        last_state, transitions_count, reward = agent.play(Environment(state_builder))
        result.write("\n".join([
            f"STEP: {step}.",
            f"Total transitions: {transitions_count}. Achieved reward: {reward}.",
            str(last_state)
        ]) + "\n" * 2)
        result.flush()
        if step > 0:
            agent.save("result")
result.close()

# %%
