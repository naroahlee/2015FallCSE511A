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
import copy

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
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
        # Initally, the Self.value was assigned as 0
        # For this problem, V_k+1(s) = max_a{Q_k(s, a)}
        # I must computer Q_0(s, a) first
        NewStateValue = util.Counter()
        WorldState = self.mdp.getStates()
        ValidAction = []

        for i in range(0, iterations):
            # print "ITER #", i, " #####################"

            for StateItem in WorldState:
                if self.mdp.isTerminal(StateItem):
                    NewStateValue[StateItem] = 0
                    continue

                ValidAction = self.mdp.getPossibleActions(StateItem)
                if 0 != len(ValidAction):
                    # print "Now Check: State", StateItem, "Action List =", ValidAction
                    # Compute the Q_k, and get the Max_a{Q_k(s, a)}
                    maxQ_k = -99999.9
                    for ActionItem in ValidAction:
                        Q_k = self.getQValue(StateItem, ActionItem)
                        # print "        Q(", StateItem, ",", ActionItem, ") = ", Q_k
                        if Q_k > maxQ_k:
                            maxQ_k = Q_k

                    NewStateValue[StateItem] = maxQ_k
            # For every i, we should update self.value.  Thus the getQValue is able to work.
            # We cannot easily using "=" for assignment.  Instead, deepcopy prevent the situation from messing up.
            self.values = copy.deepcopy(NewStateValue)

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        gamma = self.discount
        Q_k = 0
        TransList = self.mdp.getTransitionStatesAndProbs(state, action)

        for (NextState, Prob) in TransList:
            Reward = self.mdp.getReward(state, action, NextState)
            Q_k += Prob * (Reward + (gamma * self.values[NextState]))

        return Q_k


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        ValidActions = self.mdp.getPossibleActions(state)

        # Trivial Case
        if 0 == len(ValidActions) or self.mdp.isTerminal(state):
            return None

        # Policy Extraction argmax_a{Q(s, k)}
        maxQ_k = -999999.9
        maxAction = None
        for ActionItem in ValidActions:
            Q_k = self.getQValue(state, ActionItem)
            #Find the maximum action
            if Q_k > maxQ_k:
                maxQ_k = Q_k
                maxAction = ActionItem
        return maxAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
