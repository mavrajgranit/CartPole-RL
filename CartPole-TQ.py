import torch, numpy, gym, random, time
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
action_space = env.action_space.n
observation_space = env.observation_space.shape[0]
network = nn.Sequential(nn.Linear(observation_space,24),nn.ReLU(),nn.Linear(24,48),nn.ReLU(),nn.Linear(48,action_space))
target_network = nn.Sequential(nn.Linear(observation_space,24),nn.ReLU(),nn.Linear(24,48),nn.ReLU(),nn.Linear(48,action_space))
target_network.load_state_dict(network.state_dict())
criterion = nn.MSELoss()
optimizer = opt.SGD(network.parameters(),lr=0.00025)
#scheduler = opt.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.995)

epochs = 30000
max_epsilon = 1.0
min_epsilon = 0.0
percent = 0.15
decay_rate = 1/(epochs*percent)
epsilon = max_epsilon
discount_factor = 0.92
replace_epochs = 2

def preprocess(state):
    return torch.tensor(state).float()

def eps_greedy(state):
    qs = network(state)

    if random.random()>epsilon:
        q,action = qs.max(0)
        return qs,q,action.item()
    else:
        action = random.randint(0,action_space-1)
        return qs,qs[action],action

def decay():
    return max(min_epsilon,epsilon-decay_rate)

def learn(out,target):
    optimizer.zero_grad()
    loss = criterion(out,target)
    loss.backward()
    optimizer.step()
    return loss

def replace():
    target_network.load_state_dict(network.state_dict())

print("---------TRAINING---------")
mr = 0
rewardmeans = []
lossmeans = []
qvalues = []
qvals = 0
loss = 0
for e in range(epochs):
        frames = 0
        runreward = 0
        state = env.reset()
        qval = 0
        l = 0
        while True:
            frames += 1
            qs,q, action = eps_greedy(preprocess(state))
            qval += q.item()
            new_state, reward, done, i = env.step(action)
            nt = float(not done)

            target = qs.clone()
            nq, naction = target_network(preprocess(new_state)).max(0)
            target[action] = nt * discount_factor * nq + reward
            l += learn(qs, target)

            runreward+=reward
            state = new_state
            if done or frames%200 == 0:
                #print("E: "+str(e)+" F: "+str(frames)+" R: "+str(runreward)+" Eps: "+str(epsilon))
                mr += runreward
                qvals += qval/frames
                loss += l/frames
                break
        epsilon = decay()
        if (e+1)%replace_epochs==0:
            replace()
        if (e+1)%100==0:
            #scheduler.step()
            mean = mr/100
            qmean = qvals / 100
            qvalues.append(float(qmean))
            qvals = 0
            rewardmeans.append(float(mean))
            mr=0
            lossmeans.append(float(loss/100))
            loss = 0
            print(str(e + 1) + " M: " + str(mean) +" Q: "+str(qmean)+" E: " + str(epsilon))
            if mean==200.0:
                break

print("---------TESTING---------")
epsilon=0
for e in range(5):
        state = env.reset()
        qs = 0
        runreward = 0
        state = preprocess(state)
        frames=0
        while True:
            frames+=1
            qq, q, action = eps_greedy(state)
            qs+=q.item()
            new_state, reward, done, i = env.step(action)
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
#plt.savefig('./Plots/CartPole-TQ/meanloss.png',bbox_inches='tight')
plt.figure(1)
plt.plot(rewardmeans,color="orange")
plt.title("Mean Reward")
#plt.savefig('./Plots/CartPole-TQ/meanreward.png',bbox_inches='tight')
plt.figure(2)
plt.plot(qvalues,color="red")
plt.title("Mean Q-Value")
#plt.savefig('./Plots/CartPole-TQ/meanqvalue.png',bbox_inches='tight')
plt.show()