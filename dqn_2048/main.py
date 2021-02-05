import os
from pathlib import Path
from sys import argv, stdout

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from game import Direction, StateBuilder, Environment, QualityBuilder, Agent

CURRENT_PATH = Path(__file__).parent

BOARD_SIZE = 4
BOARD_UNIT = 2

GAMMA = 0.99

BATCH_SIZE = TRANSITIONS_COUNT = WARMUP_STEPS_COUNT = 5000
TARGET_SYNCING_FREQUENCY = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY_RATE = 50000

STEPS_COUNT = 200000
PLAY_EPISODES_COUNT = 100

if len(argv) < 2:
    print("Usage: python main.py <gpu_id>")
    exit()
gpu_id = argv[1]

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
    .set_optimizer(Adam(lr=1e-4))
agent = Agent(
    quality_builder,
    BATCH_SIZE, TRANSITIONS_COUNT, WARMUP_STEPS_COUNT, TARGET_SYNCING_FREQUENCY
)
agent.set_epsilons(EPSILON_START, EPSILON_END, EPSILON_DECAY_RATE)

result_path = os.path.join(CURRENT_PATH, "result")
os.makedirs(result_path, exist_ok=True)
with open(os.path.join(result_path, "log.txt"), "w+") as log_file:
    for step in range(STEPS_COUNT):
        if step % 100 == 0:
            stdout.write(f"STEP: {step}\n")
            stdout.flush()
        # Evaluate before observing next state
        if step % TARGET_SYNCING_FREQUENCY == 0:
            stdout.write("Evaluating\n")
            stdout.flush()
            total_reward = 0
            best_reward = 0
            best_state = None
            for _ in range(PLAY_EPISODES_COUNT):
                last_state, _, reward = agent.play(Environment(state_builder))
                total_reward += reward
                if best_reward < reward:
                    best_reward = reward
                    best_state = last_state
            log_file.write("\n".join([
                f"STEP: {step}. Average reward: {(total_reward / PLAY_EPISODES_COUNT):2}",
                f"Achieved reward of best episode: {best_reward}",
                str(best_state)
            ]) + "\n" * 2)
            log_file.flush()
            if step > 0:
                agent.save(result_path)
        if step == WARMUP_STEPS_COUNT:
            log_file.write("START TRAINING\n\n")
            log_file.flush()
        agent.observe(environment)
