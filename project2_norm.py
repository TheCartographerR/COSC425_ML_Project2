#
#   Ryan Pauly
#   COSC 425 - Intro to Machine Learning
#   Project 2
#
#   Part 1:
#
#   1. Make a data matrix containing just the numeric attributes that you intend to use.
#   2. Use a library SVD package to factor your data matrix and extract the singular values.
#   3. Plot a scree graph of the singular values, and plot the percentage of variance covered by the first p
#       singular values vs. p. (The variance is the square of the singular value.) What is a good choice of p?
#
#   4. Write a function to reduce your data matrix to the first p PCs, where p is the best value you have determined
#       in step (3). (Hint: To do this, use the first p columns of your V matrix. Note that the SVD package
#       returns U, Σ, and VT, since it factors X = UΣVT.)
#
#   5. Make a scatter plot of the first two PCs. You can improve your plot by annotating the points
#       with the universities numbers or names.
#
#   Part 2: Implement k-means clustering and apply it to the original data.
#
#   6. Implement a k-means clustering program. Your program should take as arguments k (the number of clusters) and
#       the data matrix to be clustered. Report the number of iterations required for convergence.
#
#   7. Report figures of merit for your clustering, including: a) the minimal intercluster distance (distance between
#       points in different clusters), b) the maximal intracluster distance (distance of distinct points within
#       a cluster), c) the Dunn index, which is the ratio of minimal intercluster
#       distance to the maximal intracluster distance (a bigger ratio is better). You can decide how to compute the
#       inter- and intracluster distances.
#
#   8. Use your program to cluster the data in (1) above (that is, clustering based on the numeric attributes
#       you chose). Experiment with different numbers of clusters and decide which best captures the structure
#       of the data.
#
#   9. Take your cluster assignments and use them to annotate, color, or otherwise
#       distinguish the clusters in your 2D scatter plot from Part 1.
#
#   10. Which other universities are in the same cluster as UTK?
#
#   11. Repeat steps (8)–(10), but cluster using a data matrix that represents the data in terms of the number p of
#       PCs you selected in step (3). Compare to your previous results. 12.Repeat step (11), but use
#       only the first two PCs.
#
#   To summarize, in steps (8)–(12.) above, you cluster based (1) on the original data, (2) on the pdimensional data,
#       and (3) on the 2-dimensional data. In each case you make a 2D scatter plot based on the first two PCs.
#
#######################################################################################################################


from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def k_means_clustering(data, k):

    # #   x_cords = reducedDataMatrix[:, 1]
    # #   y_cords = reducedDataMatrix[:, 0]

    data = data[:, :2]

    centroids = np.array(data[np.random.randint(data.shape[0], size=3), :])

    #   Original data is 55, 50
    # print("\n Shape of centroids = ", centroids.shape)
    # print("\n Centroids = ", centroids)

    oldCentroids = np.zeros(centroids.shape)       #   Store value of centroids for when we update them

    myClusters = np.zeros(len(data))

    distanceCheck = np.linalg.norm(centroids - oldCentroids, axis=1)

    iterations = 0

    while distanceCheck.any() != 0:
        iterations = iterations + 1
        #   First we assign the plots to its closest centroid to create the cluster
        for i in range(len(data)):
            distances = np.linalg.norm(data[i] - centroids, axis=1)
            newCluster = np.argmin(distances)
            myClusters[i] = newCluster

        #   Saving the new centroids.
        oldCentroids = deepcopy(centroids)
        #   Next we find the new centroids with the average value of the scatter points
        for i in range(k):
            scatter_points = [data[j] for j in range(len(data)) if myClusters[j] == i]
            if scatter_points != []:
                centroids[i] = np.mean(scatter_points, axis=0)
        #   Comparing the new with the old centroids, checking to see if its different.
        #   If centroids and old Centroids are the same we'll get 0, which means we've
        #   found the optimal centroid.
        distanceCheck = np.linalg.norm(centroids - oldCentroids, axis=1)

    scatter_points_Cluster0 = np.empty((0,2), float)
    scatter_points_Cluster1 = np.empty((0,2), float)
    scatter_points_Cluster2 = np.empty((0,2), float)

    fig, ax = plt.subplots()

    for i in range(k):
        scatter_points = np.array([data[j] for j in range(len(data)) if myClusters[j] == i])
        if scatter_points != []:
            # print("\n scatter_points = ", scatter_points.shape)

            if i == 0:
                scatter_points_Cluster0 = np.append(scatter_points_Cluster0, scatter_points, axis=0)
            if i == 1:
                scatter_points_Cluster1 = np.append(scatter_points_Cluster1, scatter_points, axis=0)
            if i == 2:
                scatter_points_Cluster2 = np.append(scatter_points_Cluster2, scatter_points, axis=0)


    # print("\n scatter_points_Cluster0 = \n", scatter_points_Cluster0)
    # print("\n scatter_points_Cluster1 = \n", scatter_points_Cluster1)
    # print("\n scatter_points_Cluster2 = \n", scatter_points_Cluster2)

    #   Find the min intercluster distance  (distance between points in different clusters)
    min_intercluster_distance = 1000000000000000000000000000000000

    for i in range(len(scatter_points_Cluster0)):
        #   Compare a scatter_points_Cluster0 point one at a time with each point in the other clusters.
        for j in range(len(scatter_points_Cluster1)):

            test_min = np.linalg.norm(abs(scatter_points_Cluster0[i] - scatter_points_Cluster1[j]))
            if test_min < min_intercluster_distance:
                min_intercluster_distance = test_min
        for j in range(len(scatter_points_Cluster2)):
            test_min = np.linalg.norm(abs(scatter_points_Cluster0[i] - scatter_points_Cluster2[j]))
            if test_min < min_intercluster_distance:
                min_intercluster_distance = test_min

    for i in range(len(scatter_points_Cluster1)):
        for j in range(len(scatter_points_Cluster2)):
            test_min = np.linalg.norm(abs(scatter_points_Cluster1[i] - scatter_points_Cluster2[j]))
            if test_min < min_intercluster_distance:
                min_intercluster_distance = test_min

    print("\n iterations to converge = ", iterations)
    print("\n min_intercluster_distance = ", min_intercluster_distance)
    #   Find the max intracluster distance  (distance of distinct points within a cluster)
    max_intracluster_distance = 0
    #   First, find the max distance within cluster0
    for i in range(len(scatter_points_Cluster0)):
        for j in range(len(scatter_points_Cluster0)):
            test_max = np.linalg.norm(abs(scatter_points_Cluster0[i] - scatter_points_Cluster0[j]))
            if test_max > max_intracluster_distance:
                max_intracluster_distance = test_max

    for i in range(len(scatter_points_Cluster1)):
        for j in range(len(scatter_points_Cluster1)):
            test_max = np.linalg.norm(abs(scatter_points_Cluster1[i] - scatter_points_Cluster1[j]))
            if test_max > max_intracluster_distance:
                max_intracluster_distance = test_max

    for i in range(len(scatter_points_Cluster2)):
        for j in range(len(scatter_points_Cluster2)):
            test_max = np.linalg.norm(abs(scatter_points_Cluster2[i] - scatter_points_Cluster2[j]))
            if test_max > max_intracluster_distance:
                max_intracluster_distance = test_max

    print("\n max_intracluster_distance = ", max_intracluster_distance)

    dunn_index = min_intercluster_distance / max_intracluster_distance
    print("\n dunn_index = ", dunn_index)

    #ax.scatter(scatter_points[:, 1], scatter_points[:, 0], c=colors[i], s=5)
    ax.scatter(scatter_points_Cluster0[:, 1], scatter_points_Cluster0[:, 0], c='r', s=5, label="Cluster 1")
    ax.scatter(scatter_points_Cluster1[:, 1], scatter_points_Cluster1[:, 0], c='g', s=5, label="Cluster 2")
    ax.scatter(scatter_points_Cluster2[:, 1], scatter_points_Cluster2[:, 0], c='b', s=5, label="Cluster 3")


    ax.scatter(centroids[:, 1], centroids[:, 0], marker='x', s=50, c='black', label="Centroids")
    plt.title("PC 1 versus PC 2: k-means Clustering", fontsize=16)
    plt.xlabel("PC2", fontsize=12)
    plt.ylabel("PC1", fontsize=12)
    plt.legend()
    plt.show()


if __name__=="__main__":

    #   PARSING and FILE INPUT:

    fileName = "UTK-peers.csv"

    attributesToAnalyze = [5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                           32, 33, 34, 35, 37, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 58, 61, 62,
                           63, 64, 65]

    # print("Number of Columns = ", len(attributesToAnalyze))

    #   DataFrame from read_csv return
    myData = pd.read_csv(fileName, usecols=attributesToAnalyze, skiprows=1, nrows=56, header=None)

    #   Remove illegal characters: '$' and ',' etc
    myData = myData.replace({'\$':'', ',':''}, regex=True)

    #   Convert empty strings to NAN
    myData = myData.replace('', np.nan).astype(float)

    myData = pd.DataFrame.dropna(myData, axis=0, how='any', thresh=None, subset=None, inplace=False)

    #   Convert all data types in the DataFrame to float
    myData = myData.astype(float)
    # print(myData)

    #   COMPUTE Z-NORMALIZATION:
    myData = (myData - myData.mean())/myData.std()

    #   Convert DataFrame to a np array in order to use numpy's SVD
    myData = myData.to_numpy()

    # print(myData)


    #    BEGIN ANALYSIS:

    U, S, V = np.linalg.svd(myData)

    # print("\n U = ", U)
    # print("\n S = ", S)
    # print("\n V = ", V)

    numOfAttributes = len(S)

    #   Plotting Variance percentage graph
    #   The variance is equivalent to the singleValues squared.
    myDataVariance = S**2
    sumOfTotalVariance = sum(myDataVariance)
    variancePercentages = []

    for i in range(numOfAttributes):
        mySummation = 0
        for j in range(i+1):
            mySummation = mySummation + S[j]**2
        variancePercentages.append(mySummation)

    variancePercentages = variancePercentages / sumOfTotalVariance * 100
    x_plotVarPercentage = list(range(1, len(variancePercentages) + 1))

    plt.plot(range(1, len(S) + 1), S, 'ro-', linewidth=2, label="Singular Values vs. # of Attributes")
    plt.xlabel("Number of Attributes", fontsize=16)
    plt.ylabel("Singular Values", fontsize=12)
    plt.title("Scree Plot", fontsize=18)
    plt.legend(loc="best")
    plt.show()

    #   UPDATE THE TITLES!
    plt.plot(x_plotVarPercentage, variancePercentages, 'ro-', linewidth=2, label="Percent Variance")
    plt.xlabel("Number of Attributes: p", fontsize=16)
    plt.ylabel("Percent of variance covered", fontsize=16)
    plt.title("Percent of Variance covered by the first p singular values vs. p", fontsize=12)
    plt.legend(loc="best")
    plt.show()

    #   A good choice of p for this data looks like somewhere between 3-5 PCs, so we'll use 4.

    first_Ps = 30

    #   Approximate with minimal number of PCs
    # print("\n Length of U = ", len(U))
    # print("\n Length of V = ", len(V))
    # print("\n S = \n", S)
    # print("\n U = \n", U)
    # print("\n S[:, :first_Ps] = \n", np.diag(S[:first_Ps]))
    # print("\n V[:first_Ps, :] = \n", V[:first_Ps, :])
    #
    # print(U.shape)
    # print(S.shape)
    # print(V.shape)

    #   Find the reduced data matrix with respect to the first_Ps variable.
    #   Part 4 of part 1:

    reducedDataMatrix = np.dot(myData, V.T[:, :first_Ps])
    # print(reducedDataMatrix)

    x_cords = reducedDataMatrix[:, 1]
    y_cords = reducedDataMatrix[:, 0]

    ax = plt.subplot
    for i in range(len(reducedDataMatrix[:, 1])):
        x = x_cords[i]
        y = y_cords[i]
        if i == 0:
            plt.scatter(x, y, c='blue', marker='o', label="PC1 vs. PC2", s=9)
        else:
            plt.scatter(x, y, c='blue', marker='o', s=9)
        plt.text(x + 0.5, y + 0.8, i, fontsize=9)
    plt.xlabel("PC2")
    plt.ylabel("PC1")
    plt.title("Reduced Data Matrix: PC1 versus PC2")
    plt.legend(loc='upper right')
    plt.show()

    x_cords = myData[:, 1]
    y_cords = myData[:, 0]

    ax = plt.subplot
    for i in range(len(myData[:, 1])):
        x = x_cords[i]
        y = y_cords[i]
        if i == 0:
            plt.scatter(x, y, c='blue', marker='o', label="PC1 vs. PC2", s=9)
        else:
            plt.scatter(x, y, c='blue', marker='o', s=9)
        plt.text(x + 0.3, y + 0.3, i, fontsize=9)
    plt.xlabel("PC2")
    plt.ylabel("PC1")
    plt.title("Original Data Matrix: PC1 versus PC2")
    plt.legend(loc='upper right')
    plt.show()

    #  PART 2:
    #   Perform k-means clustering on the reducedDataMatrix. Must count the number of iterations to convergence.

    clusters = 3

    #   Original
    k_means_clustering(myData, clusters)

    #   Two-Dimensional
    twoDimensionalData = np.dot(myData, V.T[:, 0:2])
    k_means_clustering(twoDimensionalData, clusters)

    #   p-Dimensional
    pDimensionalData = np.dot(myData, V.T[:, 0:first_Ps])
    k_means_clustering(pDimensionalData, clusters)
