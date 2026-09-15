import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Robot:
    def __init__(self, init_pose, dt=0.1):
        """
        init_pose: (x, y, theta)
        dt: time step
        """
        self.true_pose = np.array(init_pose, dtype=float)
        self.estimated_pose = np.array(init_pose, dtype=float)
        self.dt = dt
        
        # Noise parameters for odometry
        self.alpha1 = 0.05
        self.alpha2 = 0.01
        self.alpha3 = 0.05
        self.alpha4 = 0.01
        
        self.true_path = [np.copy(self.true_pose)]
        self.estimated_path = [np.copy(self.estimated_pose)]
        
    def move(self, v, w):
        """
        Moves the robot with linear velocity v and angular velocity w.
        Updates true pose and generates noisy odometry to update estimated pose.
        """
        # True kinematics update
        if w == 0:
            self.true_pose[0] += v * np.cos(self.true_pose[2]) * self.dt
            self.true_pose[1] += v * np.sin(self.true_pose[2]) * self.dt
        else:
            self.true_pose[0] += (-v/w) * np.sin(self.true_pose[2]) + (v/w) * np.sin(self.true_pose[2] + w * self.dt)
            self.true_pose[1] += (v/w) * np.cos(self.true_pose[2]) - (v/w) * np.cos(self.true_pose[2] + w * self.dt)
            self.true_pose[2] += w * self.dt
            
        self.true_pose[2] = self.normalize_angle(self.true_pose[2])
        self.true_path.append(np.copy(self.true_pose))
        
        # Noisy odometry generation
        v_hat = v + np.random.normal(0, np.sqrt(self.alpha1 * v**2 + self.alpha2 * w**2 + 1e-6))
        w_hat = w + np.random.normal(0, np.sqrt(self.alpha3 * v**2 + self.alpha4 * w**2 + 1e-6))
        
        # Odometry update
        if w_hat == 0:
            self.estimated_pose[0] += v_hat * np.cos(self.estimated_pose[2]) * self.dt
            self.estimated_pose[1] += v_hat * np.sin(self.estimated_pose[2]) * self.dt
        else:
            self.estimated_pose[0] += (-v_hat/w_hat) * np.sin(self.estimated_pose[2]) + (v_hat/w_hat) * np.sin(self.estimated_pose[2] + w_hat * self.dt)
            self.estimated_pose[1] += (v_hat/w_hat) * np.cos(self.estimated_pose[2]) - (v_hat/w_hat) * np.cos(self.estimated_pose[2] + w_hat * self.dt)
            self.estimated_pose[2] += w_hat * self.dt
            
        self.estimated_pose[2] = self.normalize_angle(self.estimated_pose[2])
        self.estimated_path.append(np.copy(self.estimated_pose))
        
        # Return odometry command for SLAM module
        return v_hat, w_hat
        
    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
        
    def get_true_pose(self):
        return self.true_pose
        
    def plot_robot(self, ax, pose, color='blue', label='Robot'):
        """Plots a simple representation of the robot."""
        x, y, theta = pose
        
        # Draw robot body
        circle = patches.Circle((x, y), 0.5, edgecolor=color, facecolor='none', label=label)
        ax.add_patch(circle)
        
        # Draw orientation line
        dx = np.cos(theta) * 0.5
        dy = np.sin(theta) * 0.5
        ax.plot([x, x+dx], [y, y+dy], color=color)
