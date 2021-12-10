from numpy import *
from matplotlib import pyplot as plt
from numpy.random import randint


class SARSA:
    # Class which implements the SARSA algorithm for a simple single player game,
    # in which an agent explores the environment, collects rewards and eventually arrives
    # in the destination state.
    # The goal is to maximize the final score (which is obtained by arriving in the
    # shortest time to the destination state), while also exploring the environment

    def __init__(self, gridSize, states_terminal, alpha, gamma_):
        # Constructor: initializes grid , valid actions ,
        # parameters and Q table
        self.gridSize = gridSize
        self.states_terminal = states_terminal
        self.alpha = alpha
        self.gamma = gamma_
        # the set of valid actions are move up, down, right, left,
        # except for the boundary walls, where only specific actions are possible
        self.valid_actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        self.max_reward = 100
        self.currentReward = -1
        self.Q = zeros((gridSize, gridSize))
        self.states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
        self.best_action = zeros((gridSize, gridSize), dtype=list)

    def generate_initial_state(self):
        # generate a random initial state in the grid
        rand_state = randint(low=0, high=len(self.states))
        initialState = self.states[rand_state]
        return initialState

    def generate_next_action(self):
        # generate a random action from the valid set of actions
        rand_action = randint(low=0, high=len(self.valid_actions))
        nextAction = self.valid_actions[rand_action]
        return nextAction

    def get_next_state(self, state, action):
        # when reaching a terminal state the max reward is achieved
        if self.is_terminal_state(state):
            return self.max_reward, state

        # the transition step is achieved by summing the current action
        # to the state, since actions are directional vectors

        # with allowed actions: [-1, 0], [1, 0], [0, 1], [0, -1]
        # we can consider the agent performing an action from
        # current location o -> x
        #
        #           |   | x |   |
        #           | x | o | x |
        #           |   | x |   |
        #

        next_state = add(state, action)

        # if the agent reaches a wall
        if -1 in next_state or self.gridSize in next_state:
            next_state = state

        return self.currentReward, next_state

    def is_terminal_state(self, state):
        # returns whether the given state is a terminal state
        for i in range(0, len(self.states_terminal)):
            check = self.states_terminal[i]
            if check[0] == state[0] and check[1] == state[1]:
                return True
        return False

    def best_neighbour_index(self, ix, iy):
        # Returns the indexes of the highest value in the Q table
        # considering the neighbourhood of a single action
        idx = -1
        idy = -1
        max_value = -Inf
        if ix != 0:
            idx, idy, max_value = self.check_max(ix - 1, iy, self.Q[ix - 1][iy], idx, idy, max_value)
        if iy != len(self.Q[ix]) - 1:
            idx, idy, max_value = self.check_max(ix, iy + 1, self.Q[ix][iy + 1], idx, idy, max_value)
        if ix != len(self.Q) - 1:
            idx, idy, max_value = self.check_max(ix + 1, iy, self.Q[ix + 1][iy], idx, idy, max_value)
        if iy != 0:
            idx, idy, max_value = self.check_max(ix, iy - 1, self.Q[ix][iy - 1], idx, idy, max_value)

        return [idx, idy]

    def search(self):
        # defines the search algorithm given the policy table
        initial_state = self.generate_initial_state()  # [x,y]
        current_state = initial_state
        it = 0
        max_it = (self.gridSize ** 2) * len(self.valid_actions)  # given maximum amount of movements
        # otherwise risk getting stuck in a loop

        while it < max_it:
            if self.is_terminal_state(current_state):
                return 1
            ix = current_state[0]
            iy = current_state[1]
            new_state = self.best_neighbour_index(ix, iy)
            # Move across the grid by selecting the highest value in the Q table

            # This value represents then the best state that can follow
            # the current one given the policy table

            best_action = subtract(new_state, current_state)

            # We can get the best action that was executed to get there by executing
            # the inverse operation than the forward one
            self.best_action[current_state[0], current_state[1]] = best_action
            current_state = new_state
            it += 1

        # Repeat the process until max movements and return 0 if the search was unsuccessful
        return 0

    def test_policy(self, num_search):
        scores = []
        # run the search algorithm for given amount of iterations
        for i in range(0, num_search):
            scores.append(self.search())

        success = [x for x in scores if x == 1]
        # collect all success
        # return probability that the search was successful
        return len(success) / num_search

    def run_iterations(self, max_iterations):
        # Improve the Q table by repeating the heuristic until eventual convergence
        for it in range(max_iterations):
            currentState = self.generate_initial_state()

            while True:
                # Get a random action
                currentAction = self.generate_next_action()
                reward, nextState = self.get_next_state(currentState, currentAction)

                # complete the stop action if the agent reached the goal state
                if reward == self.max_reward:
                    self.Q[nextState[0], nextState[1]] = self.max_reward
                    break

                cs0 = currentState[0]
                cs1 = currentState[1]
                ns0 = nextState[0]
                ns1 = nextState[1]
                # update the Q-value function Q
                self.Q[cs0, cs1] = (1 - self.alpha) * self.Q[ns0, ns1] + (
                        self.alpha * (reward + self.gamma * (self.Q[ns0, ns1])))

                # assign as current state the next state
                currentState = nextState
        return

    def log(self, Q=None):
        # Print the formatted Q table
        print('\n'.join(['\t'.join([
            '{:.4f}'.format(item)
            if (10 > item > 0) else (
                '{:.3f}'.format(item)
                if item < 100 else '{:.2f}'.format(item)
            )
            for item in row])
            for row in (self.Q if Q is None else Q)]))
        print("\n")
        return

    @staticmethod
    def check_max(ix: int, iy: int, value: float, mx: int, my: int, m_value: float):
        # if value is bigger or first call we change the index of the max value and update this max
        if value > m_value or mx == my == -1 or m_value == -Inf:
            return ix, iy, value
        else:
            return mx, my, m_value


def test_parameter(param):
    # Test the given parameter
    # Generate test range
    if param == 'alpha' or param == 'gamma':
        r = arange(0, 1, 0.01)
    elif param == 'iterations':
        r = arange(10, 400, 10)
    else:
        raise Exception(f'wrong argument param = {param}')
    p = []
    max_p = -1
    best_param = -1
    best_Q = None
    search = None
    for i in r:
        # Run search with iteration parameter
        search = SARSA(_gridSize,
                       _states_terminal,
                       i if param == 'alpha' else _alpha,
                       i if param == 'gamma' else _gamma)

        search.run_iterations(i if param == 'iterations' else _numIterations)
        # Test the policy with roll-forward search
        p_i = search.test_policy(_policy_test_range)
        # add the probability of search being successful with given parameter
        p.append(p_i)

        # update the best probability achieved
        if p_i > max_p:
            best_param = i
            max_p = p_i
            best_Q = search.Q

    # plot the iteration results
    print(f'best {param} value: {best_param} -> score:{max_p}')
    search.log(best_Q)
    plt.plot(r, p, 'o-')
    plt.title(f'Testing parameter {param}')
    plt.xlabel(f'{param}')
    plt.ylabel('Search success rate')
    plt.show()

    return p, best_param, max_p, best_Q


# default parameters

_gridSize = 4
_states_terminal = [[0, 0], [_gridSize - 1, _gridSize - 1]]
_gamma = 0.115
_alpha = 0.19
_policy_test_range = 100
_numIterations = 50

test_parameter('alpha')
test_parameter('gamma')
test_parameter('iterations')




