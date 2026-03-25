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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        val = successorGameState.getScore()

        fl = newFood.asList()
        if fl:
            dd = [manhattanDistance(newPos, f) for f in fl]
            val += 1.0 / min(dd)

        for i, gs in enumerate(newGhostStates):
            gp = gs.getPosition()
            gd = manhattanDistance(newPos, gp)
            if newScaredTimes[i] > 0:
                val += 2.0 / (gd + 0.001)
            else:
                if gd < 2:
                    val -= 9999
                elif gd < 4:
                    val -= 50.0 / gd

        return val

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def mm(st, ag, dp):
            if st.isWin() or st.isLose() or dp == 0:
                return self.evaluationFunction(st)
            na = st.getNumAgents()
            acts = st.getLegalActions(ag)
            nxt = (ag + 1) % na
            ndp = dp - 1 if ag == na - 1 else dp
            if ag == 0:
                return max(mm(st.generateSuccessor(ag, a), nxt, ndp) for a in acts)
            else:
                return min(mm(st.generateSuccessor(ag, a), nxt, ndp) for a in acts)

        na = gameState.getNumAgents()
        acts = gameState.getLegalActions(0)
        best = None
        bv = float('-inf')
        for a in acts:
            v = mm(gameState.generateSuccessor(0, a), 1 % na, self.depth if na > 1 else self.depth - 1)
            if v > bv:
                bv = v
                best = a
        return best

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def maxv(st, dp, al, be):
            if st.isWin() or st.isLose() or dp == 0:
                return self.evaluationFunction(st)
            v = float('-inf')
            for a in st.getLegalActions(0):
                v = max(v, minv(st.generateSuccessor(0, a), 1, dp, al, be))
                if v > be:
                    return v
                al = max(al, v)
            return v

        def minv(st, ag, dp, al, be):
            if st.isWin() or st.isLose():
                return self.evaluationFunction(st)
            na = st.getNumAgents()
            v = float('inf')
            for a in st.getLegalActions(ag):
                if ag == na - 1:
                    v = min(v, maxv(st.generateSuccessor(ag, a), dp - 1, al, be))
                else:
                    v = min(v, minv(st.generateSuccessor(ag, a), ag + 1, dp, al, be))
                if v < al:
                    return v
                be = min(be, v)
            return v

        al = float('-inf')
        be = float('inf')
        best = None
        bv = float('-inf')
        for a in gameState.getLegalActions(0):
            v = minv(gameState.generateSuccessor(0, a), 1, self.depth, al, be)
            if v > bv:
                bv = v
                best = a
            al = max(al, bv)
        return best

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def exmax(st, ag, dp):
            if st.isWin() or st.isLose() or dp == 0:
                return self.evaluationFunction(st)
            na = st.getNumAgents()
            acts = st.getLegalActions(ag)
            nxt = (ag + 1) % na
            ndp = dp - 1 if ag == na - 1 else dp
            if ag == 0:
                return max(exmax(st.generateSuccessor(ag, a), nxt, ndp) for a in acts)
            else:
                vs = [exmax(st.generateSuccessor(ag, a), nxt, ndp) for a in acts]
                return sum(vs) / len(vs)

        na = gameState.getNumAgents()
        acts = gameState.getLegalActions(0)
        best = None
        bv = float('-inf')
        for a in acts:
            v = exmax(gameState.generateSuccessor(0, a), 1 % na, self.depth if na > 1 else self.depth - 1)
            if v > bv:
                bv = v
                best = a
        return best

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Uses reciprocal of food distance, ghost proximity penalties/bonuses,
    capsule proximity, and raw score as features.
    """
    pos = currentGameState.getPacmanPosition()
    fd = currentGameState.getFood().asList()
    gs = currentGameState.getGhostStates()
    caps = currentGameState.getCapsules()
    sc = currentGameState.getScore()

    res = sc

    if fd:
        dists = [manhattanDistance(pos, f) for f in fd]
        res += 9.0 / min(dists)
        res -= 0.1 * len(fd)

    for g in gs:
        gp = g.getPosition()
        gd = manhattanDistance(pos, gp)
        if g.scaredTimer > 0:
            res += 200.0 / (gd + 1)
        else:
            if gd <= 1:
                res -= 600
            elif gd <= 3:
                res -= 80.0 / gd
            else:
                res -= 5.0 / gd

    for c in caps:
        cd = manhattanDistance(pos, c)
        res += 15.0 / (cd + 1)

    res -= 20 * len(caps)

    return res

# Abbreviation
better = betterEvaluationFunction
