# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    "1. Declare a stack for record my search route"

    from game import Directions
    UnCheckedState = util.Stack()
    RouteRecorderList  = []
    VisitedStateList   = []
    StartState = problem.getStartState()

    "2. Check if the start state is the goal"
    "IF We Get the Goal, then output our result"
    if True == problem.isGoalState(StartState):
        print "=== The Start State : ", StartState, " ==="
        return []

    "3. Set the Initial Pending CheckList"
    UnCheckedState.push((StartState, 0, Directions.STOP))

    while False == UnCheckedState.isEmpty():
        "Equals to go 1 step"
        NowCheckState, NowSteps, NowDirection = UnCheckedState.pop()
        "Mark the places We Visited"
        VisitedStateList.append(NowCheckState)
        "Record Our Route"
        if len(RouteRecorderList) <= NowSteps:
            RouteRecorderList.append(NowDirection)
        else:
            RouteRecorderList[NowSteps] = NowDirection

        "IF We Get the Goal, then output our Route"
        if True == problem.isGoalState(NowCheckState):
            print "=== The Goal : ", NowCheckState, " ==="
            return RouteRecorderList[1:]

        NowSuccessorList  = problem.getSuccessors(NowCheckState)
        for iState, iAction, iStepCost in NowSuccessorList:
            if iState not in VisitedStateList:
                UnCheckedState.push((iState, NowSteps + 1, iAction))

    print "=== I cannot find the way! ==="
    return []

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    import copy
    "*** YOUR CODE HERE ***"
    "1. Declare a stack for record my search route"
    UnCheckedState = util.Queue()
    "2. The Dictionary would save the VisitedStat"
    VisitedStateList   = []
    StartState = problem.getStartState()
    RouteList = []
    ActionList = []
    isSuccessful = False

    "2. Check if the start state is the goal"
    "IF We Get the Goal, then output our result"
    if True == problem.isGoalState(StartState):
        print "=== The Start State : ", StartState, " ==="
        return []

    "3. Set the Initial Pending CheckList"
    UnCheckedState.push((StartState, []))
    VisitedStateList.append(StartState)
    print "Start State", StartState
    while False == UnCheckedState.isEmpty():
        "Equals to go 1 step"
        NowCheckState, NowActionList = UnCheckedState.pop()


        "IF We Get the Goal, then output our Route"
        if True == problem.isGoalState(NowCheckState):
            isSuccessful = True
            break

        NowSuccessorList  = problem.getSuccessors(NowCheckState)
        for iState, iAction, iStepCost in NowSuccessorList:
            if iState not in VisitedStateList:
                TempActionList = copy.deepcopy(NowActionList)
                TempActionList.append(iAction)
                UnCheckedState.push((iState, TempActionList))
                VisitedStateList.append(iState)

    "==================Output======================="

    if True == isSuccessful:
        print '=== The Goal : ', NowCheckState, ' ==='
        print '===== BFS Task Over ====='
        # print NowActionList
        # return []
        return NowActionList

    else:
        print "I cannot find the answer!"
        return []

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    "1. Declare a stack for record my search route"
    import copy
    print 'Version: Rewrite for action abstaction'
    UnCheckedState = util.PriorityQueue()
    "A Dicitionary find the cost"
    VisitedStateDict   = {}
    StartState    = problem.getStartState()
    RouteList     = []
    ActionList    = []
    NowActionList = []
    isSuccessful  = False

    print '=== StartState : ', StartState, ' ==='
    "2. Check if the start state is the goal"
    "IF We Get the Goal, then output our result"
    if True == problem.isGoalState(StartState):
        print "=== I Get Goal by no move : ", StartState, " ==="
        return []

    "3. Set the Initial Pending CheckList"
    UnCheckedState.push((StartState, []), 0)
    VisitedStateDict[StartState] = 0
    cost = 0
    while False == UnCheckedState.isEmpty():
        NowCheckState, NowActionList = UnCheckedState.pop()

        "IF We Get the Goal, then output our Route"
        if True == problem.isGoalState(NowCheckState):
            isSuccessful = True
            break

        NowSuccessorList  = problem.getSuccessors(NowCheckState)
        for iState, iAction, iStepCost in NowSuccessorList:
            if False == VisitedStateDict.has_key(iState):
                NextStepCost = iStepCost + VisitedStateDict[NowCheckState]
                TempActionList = copy.deepcopy(NowActionList)
                TempActionList.append(iAction)
                UnCheckedState.push((iState, TempActionList), NextStepCost)
                VisitedStateDict[iState] = NextStepCost
            else:
                NextStepCost = iStepCost + VisitedStateDict[NowCheckState]
                if VisitedStateDict[iState] > NextStepCost:
                    TempActionList = copy.deepcopy(NowActionList)
                    TempActionList.append(iAction)
                    UnCheckedState.push((iState, TempActionList), NextStepCost)
                    VisitedStateDict[iState] = NextStepCost

    """
    Give the Result
    """

    if True == isSuccessful:
        print '=== The Goal : ', NowCheckState, ' ==='
        print '===== UCS Task Over ====='
        return NowActionList
    else:
        print '===== Fail to find the Way ====='
        return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    "1. Declare a stack for record my search route"
    import copy
    print 'Version: Astar Test Ver0'

    def creditFxn(item):
        return (item[2][0] + item[2][1])

    UnCheckedState = util.PriorityQueueWithFunction(creditFxn)
    "A Dicitionary find the cost"
    VisitedStateDict   = {}
    ProcessedList      = []
    StartState    = problem.getStartState()
    RouteList     = []
    ActionList    = []
    NowActionList = []
    isSuccessful  = False

    print '=== StartState : ', StartState, ' ==='
    "2. Check if the start state is the goal"
    "IF We Get the Goal, then output our result"
    if True == problem.isGoalState(StartState):
        print "=== I Get Goal by no move : ", StartState, " ==="
        return []

    "3. Set the Initial Pending CheckList"
    UnCheckedState.push((StartState, [], (0 , heuristic(StartState, problem))))
    VisitedStateDict[StartState] = heuristic(StartState, problem)


    while False == UnCheckedState.isEmpty():
        NowCheckState, NowActionList, NowPara = UnCheckedState.pop()
        NowCost, NowHeu = NowPara
        if NowCheckState in ProcessedList:
            continue
        "IF We Get the Goal, then output our Route"
        if True == problem.isGoalState(NowCheckState):
            isSuccessful = True
            break

        NowSuccessorList  = problem.getSuccessors(NowCheckState)
        "Dealing with the multiple push in Problem"
        ProcessedList.append(NowCheckState)
        for iState, iAction, iStepCost in NowSuccessorList:
            if False == VisitedStateDict.has_key(iState):
                NextStepCost = VisitedStateDict[NowCheckState] + iStepCost
                TempActionList = copy.deepcopy(NowActionList)
                TempActionList.append(iAction)
                UnCheckedState.push((iState, TempActionList, (NextStepCost, heuristic(iState, problem))))
                VisitedStateDict[iState] = NextStepCost
            else:
                NextStepCost = VisitedStateDict[NowCheckState] + iStepCost + heuristic(iState, problem)
                if VisitedStateDict[iState] > NextStepCost:
                    TempActionList = copy.deepcopy(NowActionList)
                    TempActionList.append(iAction)
                    UnCheckedState.push((iState, TempActionList, (NextStepCost, heuristic(iState, problem))))
                    VisitedStateDict[iState] = NextStepCost

    """
    Give the Result
    """

    if True == isSuccessful:
        print '=== The Goal : ', NowCheckState, ' ==='
        print '===== AStar Task Over ====='
        return NowActionList
    else:
        print '===== Fail to find the Way ====='
        return []



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
