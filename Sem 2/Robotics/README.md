# SLAM-Based Mapping Robot (Perception + Kinematics)

## Overview
This project implements a standalone Python simulation for Simultaneous Localization and Mapping (SLAM). Due to the constraints of not having ROS/Gazebo installed on the host system, this custom simulation fulfills the required learning outcomes by explicitly modeling robot kinematics, LiDAR sensor ray-casting, and an Occupancy Grid Mapping algorithm.

## Features
- **Differential Drive Kinematics:** Models the robot's movement and generates noisy odometry (localization error).
- **Environment and Sensor:** Simulates a 2D environment with static obstacles and casts rays to mimic a 2D LiDAR.
- **Occupancy Grid Mapping:** Updates a log-odds probabilistic grid based on sensor readings and estimated pose to create a map of the unknown environment.

## File Structure
- `environment.py`: Contains the `Environment` class that manages the 2D true map and simulates the LiDAR (`get_lidar_scan`).
- `robot.py`: Contains the `Robot` class that handles the differential drive kinematics and noisy odometry generation.
- `slam.py`: Contains the `OccupancyGridMap` class which implements the log-odds grid update using Bresenham's line algorithm.
- `main.py`: The entry point that loops through the simulation, drives the robot along waypoints, performs the SLAM updates, and visualizes the results.

## Requirements
- `uv` for environment and package management.
- Python 3.14+

## Running the Simulation
To run the simulation and see the SLAM map generated in real-time, execute:
```bash
uv run main.py
```
A window will open showing two plots:
1. **Ground Truth Environment:** Displays the true map and the true path the robot takes.
2. **Generated SLAM Map:** Displays the occupancy grid built by the SLAM algorithm, along with the noisy estimated path (odometry).

Upon completion, the script prints the Mean Squared Error (MSE) of the localization drift to the console.

## Learning Outcomes Achieved
- **Probabilistic Robotics:** Utilized log-odds formulation for the occupancy grid and Gaussian noise for the sensor/odometry models.
- **Sensor Fusion:** Combined noisy odometry (estimated pose) with LiDAR depth scans to estimate the environment structure.
- **Coordinate Frames:** Managed transformations between the local robot frame (relative angles for LiDAR) and the global map frame (Occupancy grid).
