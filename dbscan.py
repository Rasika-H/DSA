import pandas as pd
import numpy as np

# The function takes dataset, treshold distance and required number of points as input
def dbscan(D, eps, MinimumPts):
 
    # It will contain the labels assigned to each datapoint    
    labels = [0]*len(D)

    # ID of the current cluster.    
    C = 0
    
    # For each point P in the Dataset
    for P in range(0, len(D)): 
        # If point already assigned to a cluster skip it
        if not (labels[P] == 0):
           continue
        
        # Find all neighboring points of P
        NeighboringPts = region_query(D, P, eps)
        
        # If the number is below MinimumPts, this point is noise. 
        if len(NeighboringPts) < MinimumPts:
            labels[P] = -1
        # Otherwise, if there are at least MinimumPts nearby, use this point as the seed for a new cluster.   
        else: 
           C += 1
           grow_cluster(D, labels, P, NeighboringPts, C, eps, MinimumPts)
    
    # All data has been clustered!
    return labels, C


def grow_cluster(D, labels, P, NeighboringPts, C, eps, MinimumPts):

    # This function searches through the dataset to find all points that belong to this new cluster

    # Assign the cluster label to the seed point.
    labels[P] = C

    i = 0
    while i < len(NeighboringPts):    
        
        # Get the next point from the queue.        
        Pn = NeighboringPts[i]
       
        # If Pn was labelled NOISE during the seed search, then we know it's not a branch point, so make it a leaf point of cluster C and move on.
        if labels[Pn] == -1:
           labels[Pn] = C
        
        # Otherwise, if Pn isn't already claimed, claim it as part of C.
        elif labels[Pn] == 0:
            # Add Pn to cluster C
            labels[Pn] = C
            
            # Find all the neighbors of Pn
            PnNeighboringPts = region_query(D, Pn, eps)
            
            # If Pn has at least MinimumPts neighbors, its a branch point
            if len(PnNeighboringPts) >= MinimumPts:
                NeighboringPts = NeighboringPts + PnNeighboringPts           
        
        # Advance to the next point in the FIFO queue.
        i += 1


def region_query(D, P, eps):
    #This function calculates the distance between a point P and every other 
    #point in the dataset, and then returns only those points which are within a threshold distance `eps`.
    neighbors = []
    
    # For each point in the dataset...
    for Pn in range(0, len(D)):
        # If the distance is below the threshold, add it to the neighbors list.
        if np.linalg.norm(D[P] - D[Pn]) < eps:
           neighbors.append(Pn)
    return neighbors