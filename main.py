# pylint: skip-file

# %%
from os import name, environ
from sys import argv
from tensorflow import ConfigProto, Session
from keras import backend as K

if len(argv) < 2:
    print("Usage: python main.py <gpu_id>")
    exit()
GPU_ID = argv[1]

if not name == "nt":
    environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = Session(config=config)
    K.set_session(sess)

# %%
from sys import stdout
from keras import Model
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from dqn_2048 import Direction, StateBuilder, Environment, QualityBuilder, Agent

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

# %%
result = open("result.txt", "w+")
for step in range(STEPS_COUNT):
    if step % 100 == 0:
        stdout.write("STEP: {0}\n".format(step))
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
        result.write("\n".join([
            "STEP: {0}. Average reward: {1:2}".format(step, total_reward / PLAY_EPISODES_COUNT),
            "Achieved reward of best episode: {0}".format(best_reward),
            str(best_state)
        ]) + "\n" * 2)
        result.flush()
        if step > 0:
            agent.save("result")
    if step == WARMUP_STEPS_COUNT:
        result.write("START TRAINING\n\n")
        result.flush()
    agent.observe(environment)
result.close()

# %%
