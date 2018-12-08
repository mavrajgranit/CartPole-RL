import torch, numpy, gym, random, time
import torch.nn as nn
import torch.optim as opt
from torch.distributions import Categorical
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
action_space = env.action_space.n
observation_space = env.observation_space.shape[0]
network = nn.Sequential(nn.Linear(observation_space,24),nn.ReLU(),nn.Linear(24,48),nn.ReLU(),nn.Linear(48,action_space),nn.Softmax(dim=0))
criterion = nn.MSELoss()
optimizer = opt.SGD(network.parameters(),lr=0.00025)

epochs = 30000
max_epsilon = 1.0
min_epsilon = 0.0
percent = 0.2
decay_rate = 1/(epochs*percent)
epsilon = max_epsilon
discount_factor = 0.99

def preprocess(state):
    return torch.tensor(state).float()

def next_action(state):
    distribution = network(state)
    c = Categorical(distribution)
    action = c.sample()
    return action.item()

def select_action(state):
    out = network(state)
    return out.max(0)

def decay():
    return max(min_epsilon,epsilon-decay_rate)

def learn(out,target):
    optimizer.zero_grad()
    loss = criterion(out,target)
    loss.backward()
    optimizer.step()
    return loss

print("---------TRAINING---------")
mr = 0
rewardmeans = []
lossmeans = []
loss = 0
for e in range(epochs):
        frames = 0
        runreward = 0
        state = env.reset()
        state = preprocess(state)
        l = 0
        while True:
            frames += 1
            action = next_action(state)
            new_state, reward, done, i = env.step(action)
            new_state = preprocess(new_state)

            runreward+=reward
            state = new_state
            if done or frames%200 == 0:
                #print("E: "+str(e)+" F: "+str(frames)+" R: "+str(runreward)+" Eps: "+str(epsilon))
                mr += runreward
                loss += 0#l/frames
                break
        epsilon = decay()
        if (e+1)%100==0:
            mean = mr/100
            rewardmeans.append(float(mean))
            mr = 0
            lossmeans.append(float(loss / 100))
            loss = 0
            print(str(e + 1) + " M: " + str(mean) + " E: " + str(epsilon))
            if mean==200.0:
                break

print("---------TESTING---------")
epsilon=0
for e in range(5):
        state = env.reset()
        runreward = 0
        state = preprocess(state)
        frames=0
        while True:
            frames+=1
            prob ,action = select_action(state)
            new_state, reward, done, i = env.step(action.item())
            new_state = preprocess(new_state)
            env.render()
            time.sleep(0.02)
            runreward += reward
            state = new_state
            if done or frames%200 == 0:
                print("E: "+str(e)+" F: "+str(frames)+" R: "+str(runreward)+" Eps: "+str(epsilon))
                break

print("---------PLOTTING---------")
plt.figure(0)
plt.plot(lossmeans)
plt.title("Mean Loss")
plt.savefig('./Plots/PolicyGradient/meanloss.png',bbox_inches='tight')
plt.figure(1)
plt.plot(rewardmeans,color="orange")
plt.title("Mean Reward")
plt.savefig('./Plots/PolicyGradient/meanreward.png',bbox_inches='tight')
plt.show()