# DQN-Cartepole
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import random
import os
import logging
import datetime

# Disable Tensorflow Warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SEED = 511

ENV = gym.make('CartPole-v1')

# Parâmetros
EPISODES = 200  # Número de passos de treinamento
BATCH_SIZE = 24  # Experience Replay update lote
nSTATES = ENV.observation_space.shape[0]  # Espaço de estados posíveis
nACTIONS = ENV.action_space.n  # Espaço de ações possíveis
LEARNING_RATE = 0.001  # Taxa de aprendizado da rede neural
GAMA = 0.9  # Fator de desconto das recompensas (futuro => 1 | presente => 0)
EPSILON = 1  # Parametro de exploração do ambiente - Exploration / Exploitation
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995  # Taxa de Decaimento do EPSILON
MEMORY_SIZE = 500000  # Numero de elementos do buffer


class DQN:
    def __init__(self):
        self.nStates = nSTATES
        self.nActions = nACTIONS
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.lr = LEARNING_RATE
        self.gamma = GAMA
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.model = self.model()

    # Rede Neural
    def model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.nStates, activation='relu'))
        model.add(Dropout(0.05))
        model.add(Dense(24, activation='relu'))
        model.add(Dropout(0.05))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.nActions, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.lr))
        return model

    def get_action(self, state):
        # Exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nActions)  # Ação aleatória
        # Exploitation
        else:
            return np.argmax(self.model.predict(state)[0])  # Ação predita pela rede

    def experience_replay(self):

        target_list = []

        # Carrega do buffer um lote (BACTH_SIZE) de amostras aleatórias se repetição
        batch = np.array(random.sample(self.memory, BATCH_SIZE), dtype=object)

        # Define variáveis state_ e next_state_
        state_ = np.stack(batch[:, 0], axis=1)[0]
        next_state_ = np.stack(batch[:, 3], axis=1)[0]

        # Previsão da rede dos estados das amostras
        state_value = self.model.predict(state_)
        next_state_value = self.model.predict(next_state_)

        # Atualiza o target para treinamento
        index = 0
        for state, action, reward, nstate, done in batch:

            # Define o valor do target
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(next_state_value[index])

            target_f = state_value[index]
            target_f[action] = target
            target_list.append(target_f)
            index += 1

        target_ = np.array(target_list)
        epoch_count = 1

        # Treinamento da Rede Neural
        hist = self.model.fit(state_, target_, epochs=epoch_count, verbose=0)

        # Logica de decaimento do parâmetro epislon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Retorna dados do treinamento
        return hist.history['loss'][0]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# Treinamento
def train_model():
    # Controle de reprodutibilidade
    ENV.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Listas de recompensas por episódio
    rewards = []

    # Instância DQN
    dqn = DQN()

    # Preparação do Tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'TensorLog/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Iteração por episódio
    for episode in range(EPISODES):
        state = ENV.reset()
        state = np.reshape(state, [1, nSTATES])
        total_rewards = 0
        loss = np.nan
        while True:
            action = dqn.get_action(state)
            next_state, reward, done, info = ENV.step(action)
            next_state = np.reshape(next_state, [1, nSTATES])
            total_rewards += reward
            dqn.memory.append((state, action, reward, next_state, done))  # Populando dados no buffer de memória
            state = next_state

            if done or total_rewards > 199:
                rewards.append(total_rewards)
                print("episode: {}/{}, scores: {}, epsilon: {}, loss: {}".format(
                    episode, EPISODES, total_rewards, dqn.epsilon, loss))
                break

            # Experience replay
            if len(dqn.memory) > BATCH_SIZE:
                loss = dqn.experience_replay()


        # Tensorboard data
        with summary_writer.as_default():
            tf.summary.scalar('rewards', total_rewards, step=episode)
            tf.summary.scalar('loss', loss, step=episode)


        # Resolução => 100 ultimos passos com média das recompensas maior que 195
        if len(rewards) > 100 and np.average(rewards[-100:]) > 195:
            print("Resolução Alcançada!")
            break

    # Salva o modelo treinado
    dqn.save('cartpole-dqn.h5')


# Inicia treinamento
train_model()

# Avaliação do Modelo Treinado
dqn = DQN()
dqn.load('cartpole-dqn.h5')  # Carrega modelo
# Inicializa ambiente
step = 0
obs = ENV.reset()
reward_ = 0
for i in range(10000):
    step += 1
    q_values = dqn.model.predict(obs.reshape(1, -1))
    action = np.argmax(q_values[0])
    obs, reward, done, info = ENV.step(action)
    ENV.render()
    reward_ += reward
    if done:
        # print(step)
        print("\n\n\nDONE!!!!.......")
        print(reward_)
        step = 0
        reward_ = 0
        obs = ENV.reset()
