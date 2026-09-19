import heapq
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# --- A* Search Logic (Same as before) ---

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f

def heuristic(node, goal):
    return abs(node.x - goal.x) + abs(node.y - goal.y)

def get_neighbors(node, grid_size, known_obstacles):
    neighbors = []
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    for dx, dy in directions:
        new_x, new_y = node.x + dx, node.y + dy
        if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
            if (new_x, new_y) not in known_obstacles:
                neighbors.append(Node(new_x, new_y))
    return neighbors

def a_star_search(start_pos, goal_pos, grid_size, known_obstacles):
    start_node = Node(start_pos[0], start_pos[1])
    goal_node = Node(goal_pos[0], goal_pos[1])
    open_list = []
    closed_list = set()
    heapq.heappush(open_list, start_node)
    
    while open_list:
        current_node = heapq.heappop(open_list)
        if (current_node.x, current_node.y) == (goal_node.x, goal_node.y):
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]
        
        closed_list.add((current_node.x, current_node.y))
        
        for neighbor in get_neighbors(current_node, grid_size, known_obstacles):
            if (neighbor.x, neighbor.y) in closed_list: continue
            
            neighbor.g = current_node.g + 1
            neighbor.h = heuristic(neighbor, goal_node)
            neighbor.f = neighbor.g + neighbor.h
            neighbor.parent = current_node
            
            existing = next((n for n in open_list if n.x == neighbor.x and n.y == neighbor.y), None)
            if existing and existing.g <= neighbor.g: continue
            heapq.heappush(open_list, neighbor)
    return None

# --- Visualization & Environment ---

class VisualEnvironment:
    def __init__(self, size=15, obstacle_prob=0.3):
        self.size = size
        self.agent_pos = (14, 3)
        self.goal_pos = (0, 14)
        self.true_obstacles = set()
        self.known_obstacles = set()
        
        # Generate random obstacles
        for _ in range(int(size * size * obstacle_prob)):
            ox, oy = random.randint(0, size-1), random.randint(0, size-1)
            if (ox, oy) != self.agent_pos and (ox, oy) != self.goal_pos:
                self.true_obstacles.add((ox, oy))

        # Setup Plot
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title("Red: Unknown Wall | Black: Known Wall | Green: Path")
        plt.ion() # Interactive mode on

    def sense(self):
        # Reveal obstacles only when adjacent
        x, y = self.agent_pos
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in self.true_obstacles:
                self.known_obstacles.add((nx, ny))

    def update_plot(self, current_path):
        self.ax.clear()
        
        # Create a grid for visualization
        # 0 = Empty, 1 = Unknown Obstacle, 2 = Known Obstacle
        grid_data = np.zeros((self.size, self.size))
        
        # Fill grid data
        for obs in self.true_obstacles:
            grid_data[obs[1], obs[0]] = 1 # Red (Hidden)
        for obs in self.known_obstacles:
            grid_data[obs[1], obs[0]] = 2 # Black (Discovered)
            
        # Define Color Map: White, Red (faint), Black
        cmap = colors.ListedColormap(['white', '#ffcccc', 'black'])
        bounds = [0, 0.5, 1.5, 2.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Draw Grid
        self.ax.imshow(grid_data, cmap=cmap, norm=norm, origin='upper')
        
        # Draw Grid Lines
        self.ax.set_xticks(np.arange(-.5, self.size, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.size, 1), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        self.ax.tick_params(which='minor', size=0)

        # Draw Path (Green Line)
        if current_path:
            px = [p[0] for p in current_path]
            py = [p[1] for p in current_path]
            self.ax.plot(px, py, color='green', linewidth=2, alpha=0.7, label='Planned Path')

        # Draw Start, Goal, Agent
        self.ax.plot(self.goal_pos[0], self.goal_pos[1], 'r*', markersize=15, label='Goal') # Red Star
        self.ax.plot(self.agent_pos[0], self.agent_pos[1], 'bo', markersize=10, label='Agent') # Blue Dot

        plt.pause(0.2) # Pause to create animation frame

    def move(self, next_pos):
        if next_pos in self.true_obstacles:
            print("CRASH!")
            return False
        self.agent_pos = next_pos
        return True

# --- Main Loop ---

def run_visual_simulation():
    env = VisualEnvironment(size=15, obstacle_prob=0.25)
    
    print("Simulation Started. Check the popup window.")
    
    while env.agent_pos != env.goal_pos:
        # 1. Sense
        env.sense()
        
        # 2. Plan
        path = a_star_search(env.agent_pos, env.goal_pos, env.size, env.known_obstacles)
        
        # 3. Visualize
        env.update_plot(path)
        
        if not path or len(path) < 2:
            print("No path found!")
            break
            
        # 4. Act
        next_step = path[1]
        env.move(next_step)

    # Final frame
    env.update_plot([])
    print("Goal Reached!")
    plt.ioff() # Turn off interactive mode
    plt.show() # Keep window open

if __name__ == "__main__":
    run_visual_simulation()