import numpy as np
import itertools
import random
from collections import defaultdict


# Drone Rescue Environment

class DroneRescueEnv:
    """ A grid-based environment for a drone rescue mission.
    The drone must navigate to rescue targets while managing
    battery life and avoiding dangers. """

    def __init__(self, student_id: str):
        """Initialize the environment based on the student ID."""

        # Determine grid size based on last digit
        last_digit = int(student_id[-1])
        if last_digit <= 4:
            # If the student ID ends with 0–4 we have used a 5x5 grid
            self.rows, self.cols = 5, 5
            rescue_targets, charging_stations, danger_zones, blocked_cells = 2, 1, 3, 2
            wind_prob = 0.2
            max_steps = 50
        else:
            # If the student ID ends with 5-9 we have used a 6x6 grid
            self.rows, self.cols = 6, 6
            rescue_targets, charging_stations, danger_zones, blocked_cells = 3, 2, 4, 3
            wind_prob = 0.3
            max_steps = 75

        """
        Battery Configuration:
        
        If the student ID ends with an even digit then maximum battery = 10 units 
        else if the student ID ends with an odd digit then maximum battery = 15 units.
        """
        if last_digit % 2 == 0:
            max_battery = 10
        else:
            max_battery = 15

        """
        Assign cell types in the grid:
        
        S - Start position
        F - Free/Safe cell 
        D - Dangerous zone
        R - Rescue target 
        C - Charging station 
        W - Wind zone 
        X - Blocked cell / obstacle 
        A - Start position (agent's current location)
        """
        self.max_battery = max_battery
        self.wind_prob = wind_prob
        self.max_steps = max_steps
        self.grid = [['F' for _ in range(self.cols)] for _ in range(self.rows)]

        # Place start
        self.start = (0,0)
        self.grid[0][0] = 'S'

        # Random placements (for demo; you can fix deterministic positions if required)
        self.rescue_positions = self._place('R', rescue_targets)
        self.charging_positions = self._place('C', charging_stations)
        self.danger_positions = self._place('D', danger_zones)
        self.blocked_positions = self._place('X', blocked_cells)
        self.wind_positions = self._place('W', 1)

        self.reset()

    def _place(self, symbol, count):
        """ Helper function to randomly place symbols in the grid. """
        placed = []
        while len(placed) < count:
            r, c = random.randint(0,self.rows-1), random.randint(0,self.cols-1)
            if self.grid[r][c] == 'F':
                self.grid[r][c] = symbol
                placed.append((r,c))
        return placed

    def reset(self):
        """
        Reset the environment to the initial state.
        :return: The current state of the environment after reset
        """
        self.pos = self.start
        self.battery = self.max_battery
        self.steps = 0
        self.rescued = {pos: False for pos in self.rescue_positions}
        self.visited_states = set()

        return self._get_state()

    def _get_state(self):
        """Return the current state representation."""
        return self.pos, self.battery, tuple(self.rescued.values())

    def valid_actions(self):
        """Return the list of valid actions."""
        return ['UP','DOWN','LEFT','RIGHT']

    def execute_action(self, action):
        """
        Take an action and return the new state, reward, and done flag.
        The step function is responsible for Drone flying in real environment
        """
        # Update position based on action and environment dynamics
        r, c = self.pos

        # Step penalty as with each action taken, the drone consumes battery and time.
        reward = -1

        # Initialize a done flag to False, it will be set to True if the episode ends due to rescue completion or max steps reached.
        done = False

        # Increment step count
        self.steps += 1

        # Battery cost
        self.battery -= 1
        if self.battery <= 0:
            """If the battery is depleted, the drone cannot move and receives a large negative reward."""
            return self._get_state(), -20, True

        # Wind disturbance
        if self.grid[r][c] == 'W' and action != 'HOVER':
            """In a wind zone, there's a chance the drone's intended action is overridden by a random movement."""
            if random.random() < self.wind_prob:
                action = random.choice(['UP','DOWN','LEFT','RIGHT'])

        # Movement
        if action == 'UP': r -= 1
        elif action == 'DOWN': r += 1
        elif action == 'LEFT': c -= 1
        elif action == 'RIGHT': c += 1
        elif action == 'HOVER':
            if self.grid[r][c] == 'C':
                """ Hovering on a charging station restores some battery. """
                self.battery = min(self.max_battery, self.battery+2)

        # Boundary check
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            """ If the drone tries to move outside the grid, it stays in place """
            r, c = self.pos

        # Blocked cell check
        if self.grid[r][c] == 'X':
            """ If the drone tries to move into a blocked cell, it stays in place """
            r, c = self.pos

        """ Update position after movement checks """
        self.pos = (r,c)

        """ State visitation penalty to encourage exploration. """
        current_state = self._get_state()
        if current_state in self.visited_states:
            reward -= 10
        self.visited_states.add(current_state)

        # Rewards
        cell = self.grid[r][c]
        if cell == 'R' and not self.rescued[(r,c)]:
            reward += 20
            self.rescued[(r,c)] = True
        elif cell == 'D':
            reward += -10
        elif cell == 'C':
            reward += 5
            self.battery = self.max_battery

        # Termination
        if all(self.rescued.values()):
            reward = reward + 500
            done = True
        elif self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done

    def render(self):
        """Print the current state of the grid and drone status."""
        for r in range(self.rows):
            row = ""
            for c in range(self.cols):
                if (r,c) == self.pos:
                    row += "A "  # Agent's current position
                else:
                    row += self.grid[r][c] + " "
            print(row)
        print(f"Battery: {self.battery}, Rescued: {self.rescued}, Steps: {self.steps}")


class ValueIterationSolver:
    """Solver for the Drone Rescue Environment using Value Iteration."""

    def __init__(self, env, gamma=0.85, theta=1e-3):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = defaultdict(float)
        self.policy = {}

    def run(self):
        """Run value iteration until convergence."""
        iteration = 0
        while True:
            delta = 0
            for state in self._enumerate_states():
                v = self.V[state]
                self.V[state] = self._best_value(state)
                delta = max(delta, abs(v - self.V[state]))
            iteration += 1
            if delta < self.theta:
                break
        self._extract_policy()
        return iteration, delta


    def _enumerate_states(self):
        """Enumerate all possible states in the environment and return as a list."""
        states = []
        positions = [
            (r, c)
            for r in range(self.env.rows)
            for c in range(self.env.cols)
            if (r, c) not in self.env.blocked_positions
        ]
        batteries = range(1, self.env.max_battery+1)
        rescue_statuses = list(itertools.product([False,True], repeat=len(self.env.rescue_positions)))
        for pos in positions:
            for b in batteries:
                for rs in rescue_statuses:
                    states.append((pos,b,rs))
        return states

    def _best_value(self, state):
        """Calculate the best value for a given state by evaluating all possible actions."""
        best = float('-inf')

        if all(state[2]):
            return 0
        for a in self.env.valid_actions():
            val = self._evaluate_action(state,a)
            best = max(best,val)
        return best

    def _evaluate_action(self, state, action):
        """Calculate the expected return for taking an action in a given state.
        This enables the drone to mentally imagine future rewards before deciding"""


        pos, battery, rescued = state

        if battery <= 0:
            return -100

        r, c = pos # updating row number and column number of the drone

        # Simulate movement
        nr, nc = r, c

        if action == 'UP':
            nr -= 1
        elif action == 'DOWN':
            nr += 1
        elif action == 'LEFT':
            nc -= 1
        elif action == 'RIGHT':
            nc += 1

        # Boundary check
        if not (0 <= nr < self.env.rows and 0 <= nc < self.env.cols):
            nr, nc = r, c

        # Blocked cell check
        if (nr, nc) in self.env.blocked_positions:
            nr, nc = r, c

        battery -= 1

        reward = -1

        rescued_dict = dict(zip(self.env.rescue_positions, rescued))

        cell = self.env.grid[nr][nc]


        # Rescue reward
        if cell == 'R' and not rescued_dict[(nr, nc)]:
            reward += 20
            rescued_dict[(nr, nc)] = True

        # Danger penalty
        elif cell == 'D':
            reward -= 10

        # Charging station
        elif cell == 'C':

            reward += 5

            # Recharge only if battery is sufficiently low
            if battery < self.env.max_battery - 2:
                battery = self.env.max_battery

        # Penalize staying in same state
        if (nr, nc) == pos:
            reward -= 5

        # Battery depletion
        if battery <= 0:
            reward -= 20
            return reward

        # Mission complete
        if all(rescued_dict.values()):
            reward += 500
            return reward

        next_state = ((nr, nc), battery, tuple(rescued_dict.values()))

        return reward + self.gamma * self.V[next_state]

    def _extract_policy(self):
        """Extract the optimal policy from the value function."""

        for state in self._enumerate_states():
            best_action = None
            best_val = float('-inf')
            for a in self.env.valid_actions():
                val = self._evaluate_action(state,a)
                if val > best_val:
                    best_val = val
                    best_action = a
            self.policy[state] = best_action


if __name__ == "__main__":

    student_id = input("Enter your student ID: ")

    env = DroneRescueEnv(student_id)
    env.render()

    """ Value Iteration repeatedly updates the value of each state using the Bellman Optimality Equation until convergence.
    Policy Iteration has two separate phases: Policy Evaluation and Policy Improvement. 
    
    Here in this case we have chosen Value Iteration as it is more straightforward to implement 
    and often converges faster for smaller state spaces like our grid environment."""

    solver = ValueIterationSolver(env)
    iteration, delta =  solver.run()

    # Simulate following the optimal policy
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = solver.policy[state]
        next_state, reward, done = env.execute_action(action)
        total_reward += reward
        state = next_state
        env.render()

    print("\nFinal Result:")
    print(f"Converged after {iteration} iterations with delta={delta:.6f}")
    print("Total Reward Collected:", total_reward)
    print("Rescue Status:", env.rescued)
    print("Battery Remaining:", env.battery)
    print("Steps Taken:", env.steps)
