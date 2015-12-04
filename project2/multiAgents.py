# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

# Although it is not difference in Q1234 if we keep Stop in our legal list, the q6 would keeping waiting and experience score deduct and I cannot get the full score.
gIsAllowStop = False

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"

        # This Part is very tricky:
        NowFood = currentGameState.getFood()
        NowFoodList = NowFood.asList()
        if 'Stop' == action:
            return 0
        # I don't want to encounter the Ghost
        for GhostItem in newGhostStates:
            if (GhostItem.getPosition() == newPos) and (0 == GhostItem.scaredTimer):
                return 0

        # I would like to use the Manhattant Distance
        TotalDist = [];
        for fooditem in NowFoodList:
            TotalDist.append(1000000 - manhattanDistance(newPos, fooditem))

        return max(TotalDist)
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # Set the Initial Depth and Id, recursively invoke the calMiniMax
        InitDepth = 0;
        InitAgentId = 0;
        # print "self depth = ", self.depth
        # print "Agent Num  = ", gameState.getNumAgents()
        # ValMiniMax = (value, action)
        ValMiniMax = self.calMiniMax(gameState, InitDepth, InitAgentId)
        # print "FinalValue = ", ValMiniMax[0]
        return ValMiniMax[1]

    def calMiniMax(self, NowGameState, NowDepth, NowAgentId):

        # Dealing with the iteration
        NowAgentNum = NowGameState.getNumAgents()
        if NowAgentId == NowAgentNum:
            NowAgentId = 0
            NowDepth += 1

        # if we reach some leave
        if self.depth == NowDepth:
            #print "Value = ", self.evaluationFunction(NowGameState)
            return (self.evaluationFunction(NowGameState), 'Stop')

        if 0 == NowAgentId:
            return self.calMaxValue(NowGameState, NowDepth, NowAgentId)
        else:
            return self.calMinValue(NowGameState, NowDepth, NowAgentId)

    def calMaxValue(self, NowGameState, NowDepth, NowAgentId):

        MaxValue = -999999
        RecordAction = 'N/A'
        LegalActionList = NowGameState.getLegalActions(NowAgentId)
        # print "MaxLegalActionList", LegalActionList


        # Sometimes, we will reach the leaf, although the depth is not enough
        if 0 == len(LegalActionList):
            MaxValue = self.evaluationFunction(NowGameState)
            ret = (MaxValue, RecordAction)
            return ret

        for itemAction in LegalActionList:
            # Find Next Game State
            if (False == gIsAllowStop) and ('Stop' == itemAction):
                continue
            NextGameState = NowGameState.generateSuccessor(NowAgentId, itemAction)
            NextMiniMax = self.calMiniMax(NextGameState, NowDepth, NowAgentId + 1)
            NextValue = NextMiniMax[0]
            if(NextValue > MaxValue):
                MaxValue = NextValue
                RecordAction = itemAction

        ret = (MaxValue, RecordAction)
        return ret

    def calMinValue(self, NowGameState, NowDepth, NowAgentId):

        MinValue = 999999
        RecordAction = 'N/A'
        LegalActionList = NowGameState.getLegalActions(NowAgentId)
        # print "MinLegalActionList", LegalActionList

        # Sometimes, we will reach the leaf, although the depth is not enough
        if 0 == len(LegalActionList):
            MinValue = self.evaluationFunction(NowGameState)
            ret = (MinValue, RecordAction)
            return ret

        for itemAction in LegalActionList:
            # Find Next Game State
            if (False == gIsAllowStop) and ('Stop' == itemAction):
                continue
            NextGameState = NowGameState.generateSuccessor(NowAgentId, itemAction)
            NextMiniMax = self.calMiniMax(NextGameState, NowDepth, NowAgentId + 1)
            NextValue = NextMiniMax[0]
            if(NextValue < MinValue):
                MinValue = NextValue
                RecordAction = itemAction

        ret = (MinValue, RecordAction)
        return ret

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        InitDepth = 0;
        InitAgentId = 0;
        InitAlpha = -999999
        InitBeta  =  999999
        # print "self depth = ", self.depth
        # print "Agent Num  = ", gameState.getNumAgents()
        # ValMiniMax = (value, action)
        ValMiniMax = self.calMiniMaxPrune(gameState, InitDepth, InitAgentId, InitAlpha, InitBeta)
        # print "FinalValue = ", ValMiniMax[0]

        #return 'N/A'
        return ValMiniMax[1]

    def calMiniMaxPrune(self, NowGameState, NowDepth, NowAgentId, Alpha, Beta):

        # Dealing with the iteration
        NowAgentNum = NowGameState.getNumAgents()
        if NowAgentId == NowAgentNum:
            NowAgentId = 0
            NowDepth += 1

        # if we reach some leave
        if self.depth == NowDepth:
            #print "Value = ", self.evaluationFunction(NowGameState)
            return (self.evaluationFunction(NowGameState), 'Stop')

        if 0 == NowAgentId:
            return self.calMaxValuePrune(NowGameState, NowDepth, NowAgentId, Alpha, Beta)
        else:
            return self.calMinValuePrune(NowGameState, NowDepth, NowAgentId, Alpha, Beta)

    def calMaxValuePrune(self, NowGameState, NowDepth, NowAgentId, Alpha, Beta):

        MaxValue = -999999
        RecordAction = 'N/A'
        LegalActionList = NowGameState.getLegalActions(NowAgentId)
        # print "MaxLegalActionList", LegalActionList

        # Sometimes, we will reach the leaf, although the depth is not enough
        if 0 == len(LegalActionList):
            MaxValue = self.evaluationFunction(NowGameState)
            ret = (MaxValue, RecordAction)
            return ret

        for itemAction in LegalActionList:
            if (False == gIsAllowStop) and ('Stop' == itemAction):
                #print "Do not stop!  Pacman# = ", NowAgentId
                continue
            # Find Next Game State
            NextGameState = NowGameState.generateSuccessor(NowAgentId, itemAction)
            NextMiniMax = self.calMiniMaxPrune(NextGameState, NowDepth, NowAgentId + 1, Alpha, Beta)
            NextValue = NextMiniMax[0]

            # v = max(v, v(sucessor, a, b))
            if(NextValue > MaxValue):
                MaxValue = NextValue
                RecordAction = itemAction
            # if (v >= b): return v
            if(MaxValue > Beta):
                ret = (MaxValue, RecordAction)
                return ret

            if(MaxValue > Alpha):
                Alpha = MaxValue

        ret = (MaxValue, RecordAction)
        return ret

    def calMinValuePrune(self, NowGameState, NowDepth, NowAgentId, Alpha, Beta):

        MinValue = 999999
        RecordAction = 'N/A'
        LegalActionList = NowGameState.getLegalActions(NowAgentId)
        # print "MinLegalActionList", LegalActionList

        # Sometimes, we will reach the leaf, although the depth is not enough
        if 0 == len(LegalActionList):
            MinValue = self.evaluationFunction(NowGameState)
            ret = (MinValue, RecordAction)
            return ret

        for itemAction in LegalActionList:
            # Find Next Game State
            if (False == gIsAllowStop) and ('Stop' == itemAction):
                continue
            NextGameState = NowGameState.generateSuccessor(NowAgentId, itemAction)
            NextMiniMax = self.calMiniMaxPrune(NextGameState, NowDepth, NowAgentId + 1, Alpha, Beta)
            NextValue = NextMiniMax[0]

            # v = min(v, v(successor, a, b))
            if(NextValue < MinValue):
                MinValue = NextValue
                RecordAction = itemAction
            # if (v <= a): return v
            if(MinValue < Alpha):
                ret = (MinValue, RecordAction)
                return ret

            if(MinValue < Beta):
                Beta = MinValue

        ret = (MinValue, RecordAction)
        return ret

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Set the Initial Depth and Id, recursively invoke the calMiniMax
        InitDepth = 0;
        InitAgentId = 0;
        # print "self depth = ", self.depth
        # print "Agent Num  = ", gameState.getNumAgents()
        # ValMiniMax = (value, action)
        ValMiniMax = self.calExpMax(gameState, InitDepth, InitAgentId)
        # print "FinalValue = ", ValMiniMax[0]
        return ValMiniMax[1]

    def calExpMax(self, NowGameState, NowDepth, NowAgentId):

        # Dealing with the iteration
        NowAgentNum = NowGameState.getNumAgents()
        if NowAgentId == NowAgentNum:
            NowAgentId = 0
            NowDepth += 1

        # if we reach some leave
        if self.depth == NowDepth:
            #print "Value = ", self.evaluationFunction(NowGameState)
            return (self.evaluationFunction(NowGameState), 'Stop')

        if 0 == NowAgentId:
            return self.calMaxValue(NowGameState, NowDepth, NowAgentId)
        else:
            return self.calExpValue(NowGameState, NowDepth, NowAgentId)

    # I don't need to change these code since the stratagem of Pacman would not be changed at all
    def calMaxValue(self, NowGameState, NowDepth, NowAgentId):

        MaxValue = -999999
        RecordAction = 'N/A'
        LegalActionList = NowGameState.getLegalActions(NowAgentId)
        # print "MaxLegalActionList", LegalActionList

        # Sometimes, we will reach the leaf, although the depth is not enough
        if 0 == len(LegalActionList):
            MaxValue = self.evaluationFunction(NowGameState)
            ret = (MaxValue, RecordAction)
            return ret

        for itemAction in LegalActionList:
            if (False == gIsAllowStop) and ('Stop' == itemAction):
                #print "Pacman, Don't stop!  Pacman# = ", NowAgentId
                continue
            # Find Next Game State
            NextGameState = NowGameState.generateSuccessor(NowAgentId, itemAction)
            NextExpMax = self.calExpMax(NextGameState, NowDepth, NowAgentId + 1)
            NextValue = NextExpMax[0]
            if(NextValue > MaxValue):
                MaxValue = NextValue
                RecordAction = itemAction

        ret = (MaxValue, RecordAction)
        return ret


    def calExpValue(self, NowGameState, NowDepth, NowAgentId):

        MinValue = 999999
        RecordAction = 'N/A'
        LegalActionList = NowGameState.getLegalActions(NowAgentId)
        ActionLen = len(LegalActionList)
        # print "MinLegalActionList", LegalActionList

        # Sometimes, we will reach the leaf, although the depth is not enough
        if 0 == ActionLen:
            MinValue = self.evaluationFunction(NowGameState)
            ret = (MinValue, RecordAction)
            return ret

        ExpectationValue = 0
        for itemAction in LegalActionList:
            # Actually, after adding print there, I find that Ghosts NEVER stop, say, 'Stop' never emerges in LegalActionList of Ghost
            # That must be an inherit Mechanism.
            # Find Next Game State
            NextGameState = NowGameState.generateSuccessor(NowAgentId, itemAction)
            # Take Advantage of NextGameState and compute the NextExpMax
            NextExpMax = self.calExpMax(NextGameState, NowDepth, NowAgentId + 1)
            # Accumulate the value of son
            ExpectationValue += NextExpMax[0]

        # Remember to use FLOAT Type!!!
        ExpectationValue = (float)(ExpectationValue) / ActionLen
        # print 'ExpectationValue =', ExpectationValue
        # The RecordAction is non-Availible here since we don't know the exactly action the Agent takes
        # However, in order to keep the program datastructure, it is a lazy way to keep it in these pieces of code
        ret = (ExpectationValue, RecordAction)
        return ret

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      I just want to using some directly parameters, and get them with different factor.
      Then Easily computer their inner product, so that I can get a final evaluation #
      Term0: The closest food with a negative of its mannhattern distance, which I used in Question 1
      Term1: The GhostVal.  It is simliar to Term0, however, it is a simple way to avoid at least one ghost
      Term2: Score.  How can you raise another Term which is more directly than this one?
      Term3: # of Capsules.  I don't know why it is so effective.  I just tried this and it make everything great...
      Ok, then just give a little change to our Initial Factor...
      Let's commit and try.  Again again, and again.
    """
    "*** YOUR CODE HERE ***"
    Term0 = calClosetFoodVal(currentGameState)
    Term1 = calClosetGhostVal(currentGameState)
    Term2 = currentGameState.getScore()
    Term3 = len(currentGameState.getCapsules())
    Term = [Term0, Term1, Term2, Term3]
    # Factor = [1.0, 1.0, 1.0, -5.0] => 973.8
    # Factor = [1.0, 1.0, 1.0, -10.0] => 1311
    Factor = [1.0, 1.0, 1.0, -10.0]
    # Factor = [1.0, 1.0, 1.0, -50.0] => 1433
    # Factor = [1.0, 2.0, 1.0, -80.0] => 1433
    # Factor = [1.0, 1.0, 1.0, -100.0] => 1433
    Eval = sum(i*j for (i,j) in zip(Factor, Term))
    return Eval

########################### Term0 ##############################
#  Calculate the Closet Food Distance (using norm 1 == manhatten)
def calClosetFoodVal(currentGameState):
    CurPos = currentGameState.getPacmanPosition()
    NowFood = currentGameState.getFood()
    NowFoodList = NowFood.asList()

    # I would like to use the Manhattant Distance
    TotalDist = [];
    for fooditem in NowFoodList:
        TotalDist.append(-1.0 * manhattanDistance(CurPos, fooditem))
    if 0 == len(TotalDist):
        TotalDist = [0]
    ValClosetFood = max(TotalDist)
    return ValClosetFood

########################### Term1 ##############################
#  Calculate the Closet Non-Scared Ghost Distance (using norm 1 == manhatten - 1/d)
def calClosetGhostVal(currentGameState):
    ghostStatusList = currentGameState.getGhostStates()
    CurPos = currentGameState.getPacmanPosition()
    # I would like to use the Manhattant Distance
    TotalDist = [];
    for ItemGhost in ghostStatusList:
        ItemPos = ItemGhost.getPosition()
        TempDist = manhattanDistance(CurPos, ItemPos)
        if 0 == TempDist:
            TempDist = 0
        else:
            TempDist = -1.0 / TempDist
        TotalDist.append(TempDist)

    if 0 == len(TotalDist):
        TotalDist = [0]
    ValClosetGhost = min(TotalDist)
    return ValClosetGhost

# Abbreviation
better = betterEvaluationFunction



