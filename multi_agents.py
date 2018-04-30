import numpy as np
import abc
import util
from game import Agent, Action


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

        "*** YOUR CODE HERE ***"
        left_right = _left_right_score(board)
        up_down = _up_down_score(board)

        return max([left_right, up_down]) + score


def _left_right_score(board):
    """ returns an estimates score for a left or right move """
    counter = 0
    for i in range(len(board)):
        node = board[i][0]
        for j in range(len(board)):
            next_node = board[i][j]
            if node == 0 or next_node != 0 and next_node != node:
                node = next_node
            elif node == next_node and i != 0:
                counter += node*node
    return counter


def _up_down_score(board):
    """ returns an estimates score for a up or down move """
    counter = 0
    for i in range(len(board)):
        node = board[0][i]
        for j in range(len(board)):
            next_node = board[j][i]
            if node == 0 or next_node != 0 and next_node != node:
                node = next_node
            elif node == next_node and i != 0:
                counter += node*node
    return counter

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
        self.evaluation_function = util.lookup(evaluation_function, globals())
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

    class Player:
        MAX = 0
        MIN = 1

    def _minmax(self, game_state, depth, player=Player.MAX, action=None):
        """ MinMax recursive algorithm """
        if depth == 0:
            return self.evaluation_function(game_state), action
        if player == self.Player.MAX:
            best_value = (-1, None)
            for action in game_state.get_legal_actions(player):
                new_game_state = game_state.generate_successor(player, action)
                v, _ = self._minmax(new_game_state, depth, self.Player.MIN, action)
                best_value = self._get_best_value([best_value, (v, action)], player)
            return best_value
        else:
            best_value = (float('inf'), None)
            for action in game_state.get_legal_actions(player):
                new_game_state = game_state.generate_successor(player, action)
                v, _ = self._minmax(new_game_state, depth-1, self.Player.MAX, action)
                best_value = self._get_best_value([best_value, (v, action)], player)
            return best_value

    def _get_best_value(self, list_tups, player):
        """ returns the tuple with the best score depending on the player type """
        if player == MinmaxAgent.Player.MAX:
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

    class Player:
        MAX = 0
        MIN = 1

    def _alpha_beta(self, game_state, depth, alpha=(-1, None), beta=(float('inf'), None), action=None, player=Player.MAX):
        if depth == 0:
            return score_evaluation_function(game_state), action
        if player == MinmaxAgent.Player.MAX:
            v = (-1, None)
            for action in game_state.get_legal_actions(player):
                state = game_state.generate_successor(self.Player.MAX, action)
                tmp, _ = self._alpha_beta(state, depth, alpha, beta, action, self.Player.MIN)
                v = self._get_best_value([v, (tmp, action)], player)
                alpha = self._get_best_value([alpha, v], player)
                if beta[0] <= alpha[0]:
                    break
            return v
        else:
            v = (float('inf'), None)
            for action in game_state.get_legal_actions(player):
                state = game_state.generate_successor(self.Player.MIN, action)
                tmp, _ = self._alpha_beta(state, depth-1, alpha, beta, action, self.Player.MAX)
                v = self._get_best_value([v, (tmp, action)], player)
                beta = self._get_best_value([beta, v], player)
                if beta[0] <= alpha[0]:
                    break
            return v

    def _get_best_value(self, list_tups, player):
        """ returns the tuple with the best score depending on the player type """
        if player == MinmaxAgent.Player.MAX:
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
        """*** YOUR CODE HERE ***"""
        util.raiseNotDefined()

def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = better_evaluation_function
