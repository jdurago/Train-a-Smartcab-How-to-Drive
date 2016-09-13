import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

import itertools

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
    

        self.alpha = 1.0 / np.log(3) - 0.15
        self.gamma = 0.1
        self.epsilon = 0.1
        self.initialQValue = 0.0
        
        self.previousState = ('green', None, None, None, None)
        self.previousAction = None 
        self.previousReward  = 0.0
        
        self.trialNumber = 0
        
        actionList = [None, 'forward', 'left', 'right']
        lightList = ['red', 'green']
        
        # create list of all possible states
        stateList = [item for item in itertools.product(lightList, *[actionList]*4)] #tuple (light, oncoming, right, left, waypoint)

		# Initialize QTable with initialQValue values

        self.QTable = {}
        for state in stateList:
        	self.QTable[state] = {action: self.initialQValue for action in actionList}
        	
        
        	
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        
        ## Update alpha
        self.alpha = 1.0 / np.log(self.trialNumber + 3) - 0.15
        self.trialNumber += 1
        

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (inputs['light'],inputs['oncoming'], inputs['right'], inputs['left'], self.next_waypoint)
        
        # TODO: Select action according to your policy
        actionList = [None, 'forward', 'left', 'right']
        
        # select action with maximum Q value if less than epsilon, otherwise choose random action
        randNumber = np.random.uniform(low=0.0, high=1.0)
        if randNumber > self.epsilon:
        	action = max(self.QTable[self.state], key=self.QTable[self.state].get) #select action with maximum value in QTable
        else:
        	action = random.choice(actionList)
        	print "random decision was made!"
        	
        #action = self.next_waypoint # goes directly to next waypoint, without learning
        
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        
        # Update Q Table based on the following eq: Q(s,a) < - Q(s,a) + alpha[reward + gamma(Q(s', a')) - Q(s,a)]
        self.QTable[self.previousState][self.previousAction] = self.QTable[self.previousState][self.previousAction] + self.alpha * (self.previousReward  + self.gamma * (self.QTable[self.state][action]) - self.QTable[self.previousState][self.previousAction])
        
		# update states
        self.previousState = self.state
        self.previousAction = action 
        self.previousReward = reward
	
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, alpha = {}, , gamma = {}, epsilon = {}, initialQValue = {}".format(deadline, inputs, action, reward, self.alpha, self.gamma, self.epsilon, self.initialQValue)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
 	run()
				
