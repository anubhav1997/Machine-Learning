# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    val3 =0

    def setval(self, mdp, discount = 0.9, iterations = 100):
        self.val3 = iterations
        i = 0
        while(i<self.val3):
            i+=1
            
            valx = 0
            val2 = self.values.copy()
            valx += self.val3
            valx += self.val3
                
            states = self.mdp.getStates()
            
            for state in states:
                    val4 = 0
                    valx += val4
                    if self.mdp.isTerminal(state):
                        val4 =1 

                    if(val4 ==1):
                        continue

                    actions = self.mdp.getPossibleActions(state)
                    values = []
                    for j in actions: 
                      values.append(self.getQValue(state, j))
                    bestValue = max(values)

                    val2[state] = bestValue

            self.values = val2


    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        self.setval(mdp, discount, iterations)


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getqval(self, state, action):
        qVal = 0

        for state2, prob in self.mdp.getTransitionStatesAndProbs(state, action):

            reward = self.mdp.getReward(state, action, state2)
            

            temp = reward
            temp = temp + self.discount*self.values[state2]
            qVal =qVal+prob*(temp)
        
        return qVal     
    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        return self.getqval(state, action)
        
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        policies = util.Counter()
        actions = self.mdp.getPossibleActions(state)
        for action in actions:

            # how good is an action = q-value (which considers all possible outcomes)
            qval = self.getQValue(state, action)
            policies[action] = qval

        best_action = policies.argMax()
        return best_action

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
