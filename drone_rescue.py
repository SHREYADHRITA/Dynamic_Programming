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
        placed = []
        while len(placed) < count:
            r, c = random.randint(0,self.rows-1), random.randint(0,self.cols-1)
            if self.grid[r][c] == 'F':
                self.grid[r][c] = symbol
                placed.append((r,c))
        return placed

    def reset(self):
        self.pos = self.start
        self.battery = self.max_battery
        self.steps = 0
        self.rescued = {pos: False for pos in self.rescue_positions}
        return self._get_state()

    def _get_state(self):
        return (self.pos, self.battery, tuple(self.rescued.values()))

    def valid_actions(self):
        return ['UP','DOWN','LEFT','RIGHT','HOVER']

    def step(self, action):
        r, c = self.pos
        reward = -1
        done = False
        self.steps += 1

        # Battery cost
        self.battery -= 1
        if self.battery <= 0:
            return self._get_state(), -20, True

        # Wind disturbance
        if self.grid[r][c] == 'W' and action != 'HOVER':
            if random.random() < self.wind_prob:
                action = random.choice(['UP','DOWN','LEFT','RIGHT'])

        # Movement
        if action == 'UP': r -= 1
        elif action == 'DOWN': r += 1
        elif action == 'LEFT': c -= 1
        elif action == 'RIGHT': c += 1
        elif action == 'HOVER':
            if self.grid[r][c] == 'C':
                self.battery = min(self.max_battery, self.battery+2)

        # Boundary check
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            r, c = self.pos  # stay in place

        # Blocked cell check
        if self.grid[r][c] == 'X':
            r, c = self.pos  # stay in place

        self.pos = (r,c)

        # Rewards
        cell = self.grid[r][c]
        if cell == 'R' and not self.rescued[(r,c)]:
            reward += 20
            self.rescued[(r,c)] = True
            self.grid[r][c] = 'F'
        elif cell == 'D':
            reward += -10
        elif cell == 'C':
            reward += 5
            self.battery = self.max_battery

        # Termination
        if all(self.rescued.values()) or self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done

    def render(self):
        for r in range(self.rows):
            row = ""
            for c in range(self.cols):
                if (r,c) == self.pos:
                    row += "0 "
                else:
                    row += self.grid[r][c] + " "
            print(row)
        print(f"Battery: {self.battery}, Rescued: {self.rescued}, Steps: {self.steps}")


# -----------------------------
# Example Run
# -----------------------------
if __name__ == "__main__":
    student_id = input("Enter your student ID: ")
    env = DroneRescueEnv(student_id)
    env.render()
