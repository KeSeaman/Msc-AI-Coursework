import os
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from robot import Robot
from slam import OccupancyGridMap

def main():
    np.random.seed(42)
    os.makedirs('plots', exist_ok=True)
    
    # Initialize components
    env = Environment(width=50, height=50, resolution=1.0)
    init_pose = [5.0, 5.0, 0.0]
    robot = Robot(init_pose, dt=0.1)
    slam = OccupancyGridMap(width=50, height=50, resolution=1.0)
    
    # Simple predefined path (waypoints) for exploration
    # We will drive to each waypoint
    waypoints = [
        [45.0, 5.0], [45.0, 45.0], [5.0, 45.0], [5.0, 20.0],
        [30.0, 20.0], [30.0, 30.0], [15.0, 30.0], [15.0, 15.0]
    ]
    
    current_wp_idx = 0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    num_steps = 1500
    
    true_path_x = []
    true_path_y = []
    est_path_x = []
    est_path_y = []
    
    for step in range(num_steps):
        # 1. Drive robot towards waypoint
        if current_wp_idx < len(waypoints):
            tx, ty = waypoints[current_wp_idx]
            rx, ry, rtheta = robot.true_pose
            
            dx = tx - rx
            dy = ty - ry
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < 1.0:
                current_wp_idx += 1
                v, w = 0.0, 0.0
            else:
                target_theta = np.arctan2(dy, dx)
                angle_diff = robot.normalize_angle(target_theta - rtheta)
                
                # Simple proportional controller
                v = min(2.0, distance) if abs(angle_diff) < np.pi/4 else 0.0
                w = max(-1.0, min(1.0, angle_diff * 2.0))
        else:
            v, w = 0.0, 0.0
            
        # Move robot
        v_hat, w_hat = robot.move(v, w)
        
        # 2. Get LiDAR scan from True Pose
        rel_angles, distances = env.get_lidar_scan(robot.true_pose, num_rays=72)
        
        # 3. Update SLAM map using Estimated Pose (Odometry) and LiDAR scan
        # Note: In a full SLAM (like FastSLAM), we would correct the pose here using scan matching.
        # Here we just use the noisy odometry pose to show the effect of localization error on mapping.
        slam.update(robot.estimated_pose, rel_angles, distances)
        
        # Record paths
        true_path_x.append(robot.true_pose[0])
        true_path_y.append(robot.true_pose[1])
        est_path_x.append(robot.estimated_pose[0])
        est_path_y.append(robot.estimated_pose[1])
        
        if step % 50 == 0 or current_wp_idx >= len(waypoints):
            ax1.clear()
            ax2.clear()
            
            # Plot Ground Truth
            env.plot_ground_truth(ax1)
            ax1.plot(true_path_x, true_path_y, 'g-', label='True Path')
            robot.plot_robot(ax1, robot.true_pose, color='green')
            ax1.legend()
            
            # Plot Estimated Map
            map_probs = slam.get_map_probabilities()
            ax2.imshow(map_probs.T, cmap='Greys', origin='lower',
                       extent=[0, env.width, 0, env.height], vmin=0, vmax=1)
            ax2.plot(est_path_x, est_path_y, 'b-', label='Estimated Path (Odometry)')
            robot.plot_robot(ax2, robot.estimated_pose, color='blue', label='Est Robot')
            ax2.set_title("Generated SLAM Map")
            ax2.legend()
            
            plt.savefig(f'plots/step_{step:04d}.png')
            

        if current_wp_idx >= len(waypoints):
            break
            
    # Final plot and evaluation
    plt.savefig('plots/final_map.png')
    
    # Calculate MSE localization error
    true_path_arr = np.array(robot.true_path)
    est_path_arr = np.array(robot.estimated_path)
    
    mse = np.mean((true_path_arr[:, :2] - est_path_arr[:, :2])**2)
    print(f"Localization Error (MSE): {mse:.4f}")

if __name__ == '__main__':
    main()
