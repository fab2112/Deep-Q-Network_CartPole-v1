<h3 align="center">
  <img src="cartpole_example.gif" width="300">
</h3>

# CartPole-v1

O CartPole-v1 é um ambiente simulado fornecido pela *OpenAI* para treinar e
testar algoritmos de Aprendizado por Reforço. O agente é um carrinho com uma 
haste vertical que se move ao longo de um trilho sem atrito. O sistema é 
equilibrado aplicando-se uma força de +1 ou -1 empurrando-o para a 
esquerda ou para a direita. O pêndulo inicia na vertical e o objetivo é evitar 
que ele caia. Uma recompensa de +1 é fornecida para cada passo de tempo em que a 
haste permanece na posição vertical. Um episódio termina quando a haste está a 
mais de 15 graus da vertical ou o carrinho se move mais de 2,4 unidades do centro.
Após 100 passos de tempo consecutivos e uma recompensa média de 195 por episódio, o problema é 
considerado resolvido.
 
[CartPole-v1](https://gym.openai.com/envs/CartPole-v1/) 

### Arquitetura DQN

Deep-Q-Network com repetição de experiência.

[DeepReinforcementLearning](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)

### Hyperparametros DQN:

* EPISODES = 200  
* BATCH_SIZE = 24
* LEARNING_RATE = 0.001  
* GAMA = 0.9  
* EPSILON = 1 
* EPSILON_MIN = 0.01
* EPSILON_DECAY = 0.9995  
* MEMORY_SIZE = 500000 

### Rede Neural:

1. Dense layer - input: 16, output: 24, activation: **relu**
2. Dropout layer(5%)
3. Dense layer - input 24, output: 16, activation: **relu**
4. Dropout layer(5%)
5. Dense layer - input 16, output: **2**, activation: **linear**

* Função de custo => **MSE** 
* Otmizador => **Adam**
