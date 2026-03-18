# Assignment 5 – LiDAR Processing

In this assignment, LiDAR point cloud data is processed using Python. The main task was to analyse the data and apply clustering to identify structures.

## Task 1 – Ground Level

The ground level was estimated by plotting a histogram of the Z values and selecting the most frequent range.

Results:
Dataset 1:
- Ground level ≈ 61.25

Dataset 2:
- Ground level ≈ 61.27

## Task 2 – DBSCAN

To find a suitable value of eps, the elbow method was used based on k-nearest neighbour distances.

From the elbow plot, the value was chosen as:
- eps = 0.5

This value gave reasonable clustering results when DBSCAN was applied.

## Task 3 – Catenary Cluster

The largest cluster (ignoring noise) was assumed to be the catenary.  
To identify it, the X-Y span of each cluster was checked.

Results:
Dataset 1:
- min(x): 27.08
- min(y): 80.01
- max(x): 59.88
- max(y): 160.00

Dataset 2:
- min(x): 11.39
- min(y): 0.04
- max(x): 37.01
- max(y): 79.99

## Plots

The code generates:
- Histogram plots for ground detection  
- Elbow plots for eps selection  
- Cluster plots using DBSCAN  
- Catenary cluster plots  

## Final Comments
The method works well for separating structures from LiDAR data. The catenary could be identified as the largest cluster in both datasets, although results may improve further with more tuning.
