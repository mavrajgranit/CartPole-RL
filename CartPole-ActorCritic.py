import torch, numpy, gym, time
import torch.nn as nn
import torch.optim as opt
from torch.distributions import Categorical
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):

    def __init__(self,observation_space,action_space):
        super(ActorCritic,self).__init__()
        self.body1 = nn.Linear(observation_space,24)
        self.body2 = nn.Linear(24, 48)
        self.actor = nn.Linear(48,action_space)
        self.critic = nn.Linear(48,1)

    def forward(self, input):
        x = F.relu(self.body1(input))
        x = F.relu(self.body2(x))
        distribution = F.softmax(self.actor(x),dim=-1)
        q_value = self.critic(x)
        return distribution, q_value

env = gym.make("CartPole-v0")
action_space = env.action_space.n
observation_space = env.observation_space.shape[0]
#actor = nn.Sequential(nn.Linear(observation_space,24),nn.ReLU(),nn.Linear(24,48),nn.ReLU(),nn.Linear(48,action_space),nn.Softmax(dim=-1))
#critic = nn.Sequential(nn.Linear(observation_space,24),nn.ReLU(),nn.Linear(24,48),nn.ReLU(),nn.Linear(48,1))
criterion = nn.MSELoss()
#actoropt = opt.SGD(actor.parameters(),lr=0.0001) #needs to be updated slower
#criticopt = opt.SGD(critic.parameters(),lr=0.00025)

actorcritic = ActorCritic(observation_space,action_space)
optimizer = opt.SGD(actorcritic.parameters(),lr=0.00025)

epochs = 30000
discount_factor = 0.9
log_actions = torch.tensor([])
episode_rewards = []
eps =numpy.finfo(float).eps

def preprocess(state):
    return torch.tensor(state).float()

def next_action(state):
    global log_actions
    distribution,q = actorcritic(state)
    c = Categorical(distribution)
    action = c.sample()
    #actually ln
    log = torch.tensor([0.0]).add(c.log_prob(action))
    return log,action.item(),q

def select_action(state):
    out,q = actorcritic(state)
    return out.max(0)

print("---------TRAINING---------")
mr=0
rewardmeans = []
actorlossmeans = []
criticlossmeans = []
qvalues = []
qvals = 0
tdvalues = []
tdvals = 0
actloss = 0
critloss = 0
for e in range(epochs):
        frames = 0
        runreward = 0
        state = env.reset()
        state = preprocess(state)
        al = 0
        cl = 0
        qval = 0
        tdval = 0
        while True:
            frames += 1
            log, action,q = next_action(state)
            new_state, reward, done, i = env.step(action)
            new_state = preprocess(new_state)
            nt = float(not done)
            qval += q.item()

            optimizer.zero_grad()
            target = q.clone()
            _, nq = actorcritic(new_state)
            target[0] = reward + nq * discount_factor * nt
            td = target - q
            tdval += td.item()
            criticloss = (q - target) ** 2
            cl += criticloss.item()

            #Simply using the qvalue defeats the purpose since all values will be positiv and we won't be able to account everestimations
            #It should still be possible this way but seems to fail. Possibly because q-values need a very long time to predict the correct values unlike having those ready
            #I read only that the td-error is used instead. Working way better but raising questions like: What happens if the critic learns the values faster?,...
            #Training happens more slowly since the td error tends to zero
            actorloss = td.item() * log * -1
            al += actorloss.item()
            loss = criticloss+actorloss
            loss.backward()
            optimizer.step()

            '''
            q = critic(state)

            criticopt.zero_grad()
            target = q.clone()
            nq = critic(new_state)
            target[0] = reward + nq * discount_factor * nt
            td = target-q
            criticloss = (q-target)**2# criterion(q,target)
            # print(criticloss)
            criticloss.backward()
            criticopt.step()

            actoropt.zero_grad()
            actorloss = td.item()*log*-1
            actorloss.backward()
            #print(actorloss)
            actoropt.step()
            '''

            runreward+=reward
            state = new_state

            if done or frames%200 == 0:
                #print("E: "+str(e)+" F: "+str(frames)+" R: "+str(runreward))
                mr += runreward
                actloss += al / frames
                critloss += cl / frames
                qvals += qval / frames
                tdvals += tdval / frames
                break

        if (e+1)%100==0:
            qmean = float(qvals / 100)
            qvalues.append(qmean)
            qvals = 0
            tdmean = float(tdvals / 100)
            tdvalues.append(tdmean)
            tdvals = 0
            mean = float(mr/100)
            rewardmeans.append(float(mean))
            mr = 0
            actorlossmeans.append(float(actloss / 100))
            actloss = 0
            criticlossmeans.append(float(critloss / 100))
            critloss = 0
            print(str(e + 1) + " M: " + str(mean)+" Q: "+str(qmean)+" TD "+str(tdmean))
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
plt.plot(actorlossmeans)
plt.title("Actor Mean Loss per Frame")
plt.savefig('./Plots/ActorCritic/actormeanloss.png',bbox_inches='tight')
plt.figure(1)
plt.plot(criticlossmeans)
plt.title("Critic Mean Loss per Frame")
plt.savefig('./Plots/ActorCritic/criticmeanloss.png',bbox_inches='tight')
plt.figure(2)
plt.plot(qvalues,color="red")
plt.title("Mean Q Value per Frame")
plt.savefig('./Plots/ActorCritic/meanq.png',bbox_inches='tight')
plt.figure(3)
plt.plot(tdvalues,color="red")
plt.title("Mean TD Value per Frame")
plt.savefig('./Plots/ActorCritic/meantd.png',bbox_inches='tight')
plt.figure(4)
plt.plot(rewardmeans, color="orange")
plt.title("Mean Reward per 100 Episodes")
plt.savefig('./Plots/ActorCritic/meanreward.png',bbox_inches='tight')
plt.show()