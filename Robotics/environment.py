import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, width=50, height=50, resolution=1.0):
        self.width = width
        self.height = height
        self.resolution = resolution
        
        # Ground truth grid: 0 is free, 1 is obstacle
        self.grid_x_size = int(width / resolution)
        self.grid_y_size = int(height / resolution)
        self.grid = np.zeros((self.grid_x_size, self.grid_y_size))
        
        self.generate_obstacles()
        
    def generate_obstacles(self):
        """Generates some random obstacles in the environment."""
        # Outer walls
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        
        # Add some inner walls / boxes
        self.grid[10:15, 10:30] = 1
        self.grid[30:45, 10:15] = 1
        self.grid[25:30, 35:45] = 1
        
        # Add a few random blocks
        for _ in range(5):
            cx = np.random.randint(5, self.grid_x_size - 10)
            cy = np.random.randint(5, self.grid_y_size - 10)
            w = np.random.randint(2, 6)
            h = np.random.randint(2, 6)
            self.grid[cx:cx+w, cy:cy+h] = 1
            
    def get_lidar_scan(self, pose, max_range=15.0, num_rays=36, fov=np.pi*2, noise_std=0.05):
        """
        Simulates a 2D LiDAR scan using ray casting.
        pose: (x, y, theta)
        """
        x, y, theta = pose
        rel_angles = np.linspace(-fov/2, fov/2, num_rays, endpoint=False)
        distances = np.full(num_rays, max_range)
        
        step_size = self.resolution * 0.5  # Step size for ray casting
        
        for i, rel_angle in enumerate(rel_angles):
            angle = theta + rel_angle
            dx = np.cos(angle) * step_size
            dy = np.sin(angle) * step_size
            
            rx, ry = x, y
            dist = 0.0
            
            while dist < max_range:
                rx += dx
                ry += dy
                dist += step_size
                
                # Check grid coordinates
                gx = int(rx / self.resolution)
                gy = int(ry / self.resolution)
                
                if gx < 0 or gx >= self.grid_x_size or gy < 0 or gy >= self.grid_y_size:
                    distances[i] = dist
                    break
                    
                if self.grid[gx, gy] == 1:
                    # Hit an obstacle
                    distances[i] = dist + np.random.normal(0, noise_std)
                    break
                    
        return rel_angles, distances
        
    def plot_ground_truth(self, ax):
        """Plots the ground truth grid."""
        ax.imshow(self.grid.T, cmap='Greys', origin='lower', 
                  extent=[0, self.width, 0, self.height], alpha=0.5)
        ax.set_title("Ground Truth Environment")
