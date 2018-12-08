import numpy, gym, random, time
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
action_space = env.action_space.n
observation_space = 10**env.observation_space.shape[0]
qtable = numpy.zeros((observation_space,action_space))

epochs = 70000
max_epsilon = 1.0
min_epsilon = 0.0
percent = 0.3
decay_rate = 1/(epochs*percent)
epsilon = max_epsilon
discount_factor = 0.99
lr = 0.6

#10^4 states are supported
#Each input is mapped to a range of 0-9
#According to that the final state is constructed as x1*10^0 + x2*10^1 + x3*10^2 + x4*10^3 -> 0000-9999

def map(inputlow,inputhigh,value,outputlow,outputhigh):
    inputpercentage = (value-inputlow)/(inputhigh-inputlow)
    output = outputhigh*inputpercentage-outputlow*inputpercentage+outputlow
    return output

#State consists of(taken from https://github.com/openai/gym/wiki/CartPole-v0):
#0 	Cart Position 	-2.4 	2.4
#1 	Cart Velocity 	-Inf 	Inf
#2 	Pole Angle 	~ -41.8° 	~ 41.8°
#3 	Pole Velocity At Tip 	-Inf 	Inf

#More common values(idea from https://github.com/openai/gym/issues/989):
#Values have been adjusted
#Use this to determine correct boundaries: import sys
def preprocess(state):
    pos = round(map(-2.41,2.41,state[0],0,9))#map(-2.4,2.4,state[0],0,9)
    #if pos>9:
    #    sys.exit("Pos to high "+str(state[0]))
    pos = max(0,min(9,pos))
    vel = round(map(-3.41,3.41,state[1],0,9))#map(-1,1,state[1],0,9)
    #if vel>9:
    #    sys.exit("Vel to high "+str(state[1]))
    vel = max(0, min(9, vel))
    angle = round(map(-0.28,0.28,state[2],0,9))#map(-41.8,41.8,state[2],0,9)
    #if angle>9:
    #    sys.exit("Angle to high "+str(state[2]))
    angle = max(0, min(9, angle))
    pole_vel = round(map(-3.5,3.5,state[3],0,9))#map(-1,1,state[3],0,9)
    #if pole_vel>9:
    #    sys.exit("Polevel to high "+str(state[3]))
    pole_vel = max(0, min(9, pole_vel))
    prepstate = int(pos+vel*10+angle*100+pole_vel*1000)
    return prepstate

def eps_greedy(state):
    qs = qtable[state]

    if random.random()>epsilon:
        action = numpy.argmax(qs)
        q = qs[action]
        return qs,q,action.item()
    else:
        action = random.randint(0,action_space-1)
        return qs,qs[action],action

def maxq(state):
    qs = qtable[state]
    return numpy.max(qs)

def decay():
    return max(min_epsilon,epsilon-decay_rate)

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
        state = preprocess(state)
        qval = 0
        l = 0
        while True:
            frames += 1
            qs,q, action = eps_greedy(state)
            qval += q
            new_state, reward, done, i = env.step(action)
            new_state = preprocess(new_state)
            nt = float(not done)

            nq = maxq(new_state)
            newq = nt * discount_factor * nq + reward
            nval = (1-lr)*q + lr*newq
            loss = ((newq-q)**2)/2
            qtable[state][action] = nval
            l += loss

            runreward+=reward
            state = new_state
            if done or frames%200 == 0:
                #print("E: "+str(e)+" F: "+str(frames)+" R: "+str(runreward)+" Eps: "+str(epsilon))
                mr += runreward
                qvals += qval/frames
                loss += l/frames
                break
        epsilon = decay()
        if (e+1)%100==0:
            mean = mr/100
            qmean = qvals / 100
            qvalues.append(float(qmean))
            qvals = 0
            rewardmeans.append(float(mean))
            mr = 0
            lossmeans.append(float(loss / 100))
            loss = 0
            print(str(e + 1) + " M: " + str(mean) +" Q: "+str(qmean) + " E: " + str(epsilon))
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
            qs+=q
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
#plt.savefig('./Plots/CartPole-QTable/meanloss.png',bbox_inches='tight')
plt.figure(1)
plt.plot(rewardmeans,color="orange")
plt.title("Mean Reward")
#plt.savefig('./Plots/CartPole-QTable/meanreward.png',bbox_inches='tight')
plt.figure(2)
plt.plot(qvalues,color="red")
plt.title("Mean Q-Value")
#plt.savefig('./Plots/CartPole-QTable/meanqvalue.png',bbox_inches='tight')
plt.show()