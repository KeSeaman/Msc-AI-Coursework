import numpy as np
import math

class OccupancyGridMap:
    def __init__(self, width=50, height=50, resolution=1.0):
        self.width = width
        self.height = height
        self.resolution = resolution
        
        self.grid_x_size = int(width / resolution)
        self.grid_y_size = int(height / resolution)
        
        # Log odds map initialized to 0 (unknown)
        self.log_odds = np.zeros((self.grid_x_size, self.grid_y_size))
        
        # Log odds constants
        self.l_occ = np.log(0.7 / 0.3)
        self.l_free = np.log(0.3 / 0.7)
        self.l_0 = 0.0
        
    def update(self, pose, angles, distances, max_range=15.0):
        """
        Updates the occupancy grid given a pose and a LiDAR scan.
        pose: (x, y, theta)
        """
        x, y, theta = pose
        
        for i, angle in enumerate(angles):
            dist = distances[i]
            
            # Global angle of the ray
            ray_angle = theta + angle
            
            # Start and end points of the ray
            x0, y0 = x, y
            
            # If the ray hit nothing, we consider it free up to max_range
            is_hit = dist < max_range
            
            # Endpoint
            x1 = x + dist * np.cos(ray_angle)
            y1 = y + dist * np.sin(ray_angle)
            
            # Bresenham's line algorithm to find all cells crossed by the ray
            cells = list(self.bresenham(x0, y0, x1, y1))
            
            # Update cells
            for i, (cx, cy) in enumerate(cells):
                if 0 <= cx < self.grid_x_size and 0 <= cy < self.grid_y_size:
                    if i == len(cells) - 1 and is_hit:
                        # Endpoint is occupied
                        self.log_odds[cx, cy] += self.l_occ
                    else:
                        # Intermediate points are free
                        self.log_odds[cx, cy] += self.l_free
                        
    def bresenham(self, x0, y0, x1, y1):
        """Yields integer coordinates crossed by a line from (x0, y0) to (x1, y1)."""
        x0 = int(x0 / self.resolution)
        y0 = int(y0 / self.resolution)
        x1 = int(x1 / self.resolution)
        y1 = int(y1 / self.resolution)
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                yield (x, y)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                yield (x, y)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        yield (x, y)
        
    def get_map_probabilities(self):
        """Converts log odds to probabilities (0 to 1)."""
        return 1.0 - (1.0 / (1.0 + np.exp(self.log_odds)))
