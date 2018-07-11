# AI for Self Driving Car

# Importing the libraries

import numpy as np
# work with array 
import random
# some random samples
import os
# load the model and save the model
import torch
# n-network dyanamic Graphs
import torch.nn as nn
# n-network 
import torch.nn.functional as F
# fucntional package for nn module loss function uber loss 
import torch.optim as optim
# used to optimizers sochastic gradient decent
import torch.autograd as autograd
# Variable calss from autograd 
from torch.autograd import Variable
# we import Variable 

# Creating the architecture of the Neural Network

#inherit from nn.Module
class Network(nn.Module):
    # self is allways there, input_size = no of input neuron,nb_action is no of output neurons
    # in our case is 5 and actions 3 
    def __init__(self, input_size, nb_action):
        #super used to get the functions into the self instance of the class
        super(Network, self).__init__()
        # attact to the object variables the parameters.
        self.input_size = input_size
        self.nb_action = nb_action
        # finally we use the linear functions to make the connections 
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
        
        #self to us the variables , inputs states are the actuall states where input_size was the number of states.
    def forward(self, state):
        # F is the functional module of the neural network, relu is retifier
        # we input the sate into the first full connection 
        # we use self as its part of the instance of the object 
        x = F.relu(self.fc1(state))
        # finall we output neurons which is the Q_values 
        q_values = self.fc2(x)
        #finally we output the q_values 
        return q_values
    
        #try yourself !!!!!

# Implementing Experience Replay
# like we learned in the theory classes 

class ReplayMemory(object):
    
    # The size of the memory which is caplcity and the memory itself.
    def __init__(self, capacity):
        #here we have capacity which is the capacity of the whole expreiance replay in 
        #in our case we are choosing this as 100,000
        self.capacity = capacity
        # we make a memory array that will hold all these experinces size of which is defined 
        #by capacity.
        self.memory = []
        
    # never more than 100,000 size of memory.or basically never greater than capacity 
    def push(self, event):
    # form of event st,st+1,At,last reward.
    # we use th append function of append an new event to the memory 
        self.memory.append(event)
        #len is a function of python that return the lengtj of array 
        if len(self.memory) > self.capacity:
            # del means delete to remove elements from array  
            del self.memory[0]
            
    #take a few samples of the memory 
    def sample(self, batch_size):
        # the batch size usually 100 for our example 
        # first we take a random sample using the random class we imported. 
        #batch size specifices the size of this random batch,
        # the zip function will be explained in detail in class please look into notes notes 
        samples = zip(*random.sample(self.memory, batch_size))
        # first we need to understand what torch.cat does. 
        # it turns this normal array intp a torch varriable. 
        # now we rap in with a Variable class for gradient decent.
        # why are we using Map then well that's how we go through an array a shortcut to 
        # the for loop 
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning
        
# this is the Dqn we call in the map to run the AI
class Dqn():
    
    # we need input_size and nb_action as we make an object of the network calss 
    # with will be 5 and 3 in our case.
    # and we have gamma from our theory calss. discount factor.
    def __init__(self, input_size, nb_action, gamma):
        # basic attacting varriable to object 
        self.gamma = gamma
        # sliding window of the mean of the last 100 rewards to calculate the. 
        # effectivness and how ai is learning.
        self.reward_window = []
        # now we creat the neural network. Object of the network class.
        self.model = Network(input_size, nb_action)
        # now we creat memory, which is the object of the ReplayMemory class
        #like discussed eariler we are using 100,000
        self.memory = ReplayMemory(100000)
        # Adam is a very good optimizer 
        # we send it the parameter of our model.
        # and we keep a learning rate 
        # if its equal to one its a bad idea and will not learn at all.
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        # we need to conver the vector into a tensor and then add another dimention for 
        #batch 
        # we conver it into a Tensor using torch.Tensor of size input_size and add a dimention using unsqueeze
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        # initally both action and reward is zero 
        self.last_action = 0
        self.last_reward = 0
    # selects the action to perform 0,1,2.
    # from the output of the nureal network ,action depends on the input state.
    # we will put the sate into the netowrk called model and get the qvalues in return 
    def select_action(self, state):
        # softmax helps us explore the enviromnet a little bit using temprature 
        #it give back probablity 
        #we get the values for the softmax from the model which gives back the q values 
        # we get the Qvalue as when we give the model a Variable packed sate 
        # which convest tensor to gadent 
        # its volatile because we Wont need its gradient later on
        # temprature is multiplied to the qvalues 
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        # finally random draw of the probablity distribution we jsut obtained 
        action = probs.multinomial()
        # finally get the output at [0,0]
        return action.data[0,0]
    
    # lest start traning the deep nural network 
    # state, next_state,reward,action are the elements of a MDP
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # we take the batches from memory
        # we first take the model with batch current state and get the actions for that sates thus this is the 
        # action it took right now 
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # this was the action it took last time and more exatly its the maximum of the qvalues.
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        # now we find the target by multiplying Gamma with next_output and adding the batch reward.
        target = self.gamma*next_outputs + batch_reward
        # now we calculte the loss using the funtion we saw before.
        td_loss = F.smooth_l1_loss(outputs, target)
        #we need to restrat it at each loop ittration thus the line bellow 
        # which is the zero_grad funtion 
        self.optimizer.zero_grad()
        # we backpropogate it using the .backward function
        td_loss.backward(retain_variables = True)
        # this just updates the weights after backpropogation.
        self.optimizer.step()
    
    # updates the value of last_action,last_state,last_reward
    # line 131 in map
    # its acutally last reward and last signal 
    def update(self, reward, new_signal):
        # new state depends on the new_signal 
        # as its a array we make it a tensor 
        # array of signal 1, signal 2, signal 3.
        # unsqueeze for batch.
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        # now we update the memory 
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        # we use tensors for all other so we make last action and last reward tensors too
        #use long tensor for numbers ,mainly integer into tensor. we put it in a bracket to make an array 
        
        #this is the new ittration of the ai Run.
        # now we get the action to be performed using the new state.
        action = self.select_action(new_state)
        
        #if memory have more than 100 elements then do a sampling 
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            # make the Ai lean using this batch.
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        # update the reward window if grater than 1000 then delete the first element 
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        # return action which will be used in the map.py
        return action
    
    # Computes the score. 
    def score(self):
        # basically an average.
        return sum(self.reward_window)/(len(self.reward_window)+1.)
        
        #saves the brain / model.
    def save(self):
        # this is a python dictionary 
        # mode , optimizer 
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    # this used to load the model back in. 
    # os is used to find the path. 
    # is path return true if file exsist.
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")