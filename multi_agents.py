import numpy as np
import abc
import util
from game import Agent, Action

# global definitions ##########################################

MAX_PLAYER = 0
MIN_PLAYER = 1
INFINITY = float('inf')
MIN_INFINITY = float('-inf')
NP_TABLE = np.asarray([[-0.79062704, -0.68341923, -0.78454538, -0.16428198],
                       [-0.82002277, -0.58707786, -0.30676488, -0.19649091],
                       [-0.86824234, -0.66063053, -0.368416  , -0.21255646],
                       [-0.86953906, -0.51801351, -0.28521974, -0.05791075]], dtype=np.float)

# helper functions ############################################


def _get_best_value(list_tups, player):
    """ returns the tuple with the best score depending on the player type
        @param list_tups a list of tuples [(score, action), (score, action), ...]
        @param player current player
        @return the tuple with the best value according to the given player """
    if player == MAX_PLAYER:
        best = list_tups[0]
        for tup in list_tups:
            if tup[0] > best[0]:
                best = tup
        return best
    else:
        best = list_tups[0]
        for tup in list_tups:
            if tup[0] < best[0]:
                best = tup
        return best


def _left_right(board):
    """ return an estimated score for a left or right move """
    estimated_score = 0
    board = board.copy()
    for i in range(len(board)):
        for j in range(len(board)):
            node = board[i][j]
            if node == 0:
                continue
            for k in range(j, len(board)):
                curr = board[i][k]
                if j == k:
                    continue
                if node == curr:
                    estimated_score += node*2
                    board[i][j] = 0
                    board[i][k] = 0

    return estimated_score


def _up_down(board):
    """ return an estimated score for an up or down move """
    estimated_score = 0
    board = board.copy()
    for i in range(len(board)):
        for j in range(len(board)):
            node = board[j][i]
            if node == 0:
                continue
            for k in range(j, len(board)):
                curr = board[k][i]
                if j == k:
                    continue
                if node == curr:
                    estimated_score += node*2
                    board[j][i] = 0
                    board[k][i] = 0
    return estimated_score

# end of helper functions ############################################


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        score = successor_game_state.score

        left_right = _left_right(board)
        up_down = _up_down(board)

        return max([left_right, up_down]) + score


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        if type(evaluation_function) is str:
            self.evaluation_function = util.lookup(evaluation_function, globals())
        else:
            self.evaluation_function = evaluation_function
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        _, best_action = self._minmax(game_state, self.depth)
        if best_action is None:
            best_action = Action.STOP
        return best_action

    def _minmax(self, game_state, depth, player=MAX_PLAYER, action=None):
        """ MinMax recursive algorithm """
        if depth == 0:
            return self.evaluation_function(game_state), action
        if player == MAX_PLAYER:
            best_value = (MIN_INFINITY, None)
            for action in game_state.get_legal_actions(player):
                new_game_state = game_state.generate_successor(player, action)
                v, _ = self._minmax(new_game_state, depth, MIN_PLAYER, action)
                if v != INFINITY:
                    best_value = _get_best_value([best_value, (v, action)], player)
            return best_value
        else:
            best_value = (INFINITY, None)
            for action in game_state.get_legal_actions(player):
                new_game_state = game_state.generate_successor(player, action)
                v, _ = self._minmax(new_game_state, depth-1, MAX_PLAYER, action)
                if v != MIN_INFINITY:
                    best_value = _get_best_value([best_value, (v, action)], player)
            return best_value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        _, action = self._alpha_beta(game_state, self.depth)
        if action is None:
            action = Action.STOP
        return action

    def _alpha_beta(self, game_state, depth, alpha=(MIN_INFINITY, None), beta=(INFINITY, None), action=None, player=MAX_PLAYER):
        if depth == 0:
            return self.evaluation_function(game_state), action
        if player == MAX_PLAYER:
            v = (MIN_INFINITY, None)
            for action in game_state.get_legal_actions(player):
                state = game_state.generate_successor(MAX_PLAYER, action)
                tmp, _ = self._alpha_beta(state, depth, alpha, beta, action, MIN_PLAYER)
                if tmp != INFINITY:
                    v = _get_best_value([v, (tmp, action)], player)
                    alpha = _get_best_value([alpha, v], player)
                if beta[0] <= alpha[0]:
                    break
            return v
        else:
            v = (INFINITY, None)
            for action in game_state.get_legal_actions(player):
                state = game_state.generate_successor(player, action)
                tmp, _ = self._alpha_beta(state, depth-1, alpha, beta, action, MAX_PLAYER)
                if tmp != MIN_INFINITY:
                    v = _get_best_value([v, (tmp, action)], player)
                    beta = _get_best_value([beta, v], player)
                if beta[0] <= alpha[0]:
                    break
            return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        _, best_action = self._expectimax(game_state, self.depth)
        if best_action is None:
            best_action = Action.STOP
        return best_action

    def _expectimax(self, game_state, depth, player=MAX_PLAYER, action=None):
        """ Expectimax recursive algorithm """
        if depth == 0:
            return self.evaluation_function(game_state), action
        if player == MAX_PLAYER:
            best_value = (MIN_INFINITY, None)
            for action in game_state.get_legal_actions(player):
                new_game_state = game_state.generate_successor(player, action)
                v, _ = self._expectimax(new_game_state, depth, MIN_PLAYER, action)
                if v != INFINITY:
                    best_value = _get_best_value([best_value, (v, action)], player)
            return best_value
        else:
            values = list()
            for action in game_state.get_legal_actions(player):
                new_game_state = game_state.generate_successor(player, action)
                v, _ = self._expectimax(new_game_state, depth-1, MAX_PLAYER, action)
                if v != MIN_INFINITY:
                    values.append(v)
            if len(values) == 0:
                expected = INFINITY
            else:
                probability = 1/len(values)
                expected = sum(map(lambda x: x*probability, values))
            return expected, None


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: mutiplies a cost table with the game baord and sums the values.
    The table was generated by a generic algorithm that ran on a depth 1 game with AlphaBetaAgent
    On a population of 160 over 144 generations.
    The fitness of each table was score/total_population_score
    Crossover was determined by the relations of the parents scores
    Mutations were made at a low probability of 0.002 for each cell
    The best table was selected (found at generation 139)
    """

    board = current_game_state.board

    score = np.sum(np.sum(board*NP_TABLE))
    return score


# Abbreviation
better = better_evaluation_function
