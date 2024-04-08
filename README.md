# ICP_Implementation


The code below works with point clouds. Point clouds are data that can be extracted in different ways: from a pair of cameras (stereo geometry), LiDAR (Light Detection And Ranging) sensors, depth sensors, etc. With a set of 30 scans extracted from a LiDAR, this data is part of the KITTI DATASET. Using the point clouds from each scan, it estimates the final trajectory of the vehicle, starting from the first scan. To estimate the car's trajectory, the Iterative Closest Points (ICP) algorithm was used. The ground-truth is a .npy file, which can be opened using the NumPy library. When loaded, you will have an array of size (30, 4, 4). Each row, the first index, represents a transformation matrix in homogeneous coordinates for each of the 30 positions of the car.
