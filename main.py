# pylint: skip-file

#%%
"""
Main
"""

from dqn_2048 import Direction, StateBuilder, ActionBuilder, Environment, QualityBuilder, Agent

BOARD_SIZE = 4
BOARD_UNIT = 2

GAMMA = 0.99
LEARNING_RATE = 1e-4

BATCH_SIZE = 20
EXPERIENCE_COUNTS = STARTING_STEP = 10 ** 3
TARGET_SYNCING_FREQUENCY = 10 ** 2
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY_RATE = 10 ** 4

STEPS_COUNT = 10 ** 5

state_builder = StateBuilder().set_size(BOARD_SIZE).set_unit(BOARD_UNIT)
action_buider = ActionBuilder()
environment = Environment(state_builder)

quality_builder = QualityBuilder().set_gamma(GAMMA) \
    .set_input_size(BOARD_SIZE ** 2) \
    .set_output_size(len(Direction)) \
    .set_learning_rate(LEARNING_RATE)
agent = Agent(
    quality_builder, action_buider,
    BATCH_SIZE, EXPERIENCE_COUNTS, STARTING_STEP, TARGET_SYNCING_FREQUENCY,
    EPSILON_START, EPSILON_END, EPSILON_DECAY_RATE
)

result = open("result.txt", "w+")
for step in range(STEPS_COUNT):
    agent.observe(environment)
    if step % (STEPS_COUNT / 10) == 0:
        (
            last_state, total_actions_count, wasted_actions_count, reward
        ) = agent.play(Environment(state_builder))
        result.write("\n".join([
            f"STEP: {step}.",
            f"Total actions: {total_actions_count}. Wasted actions: {wasted_actions_count}.",
            f"Achieved reward: {reward}.",
            str(last_state)
        ]) + "\n" * 2)
        result.flush()
result.close()

# %%
