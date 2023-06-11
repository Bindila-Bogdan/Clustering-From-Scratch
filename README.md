# Clustering from Scratch

### Experimental project that contains four clustering algorithms which can be applied on tabular data

This project represents a playground for testing, optimizing, comparing and visualizing four clustering algorithms. It includes KMeans, DBSCAN, Gaussian Mixture and Agglomerative Clustering methods that are implemented in Python. Additionaly, there are implemented various distance metrics and clustering scores for assesing the quality of the results. 

## Features

- Two main modes:
  - optimizes each algorithm for one data set and picks the best one
  - runs each algorithm on one data set and display the results
- Includes distances matrices like: Euclidean, Manhattan, Cosine similarity an Pearson Correlation
- Embeds the automated optimization process fo each algorithm
- Implemented quality scores: WSS, Rand Index and Purity
- Includes data processing and dimensionality reduction for visualization
- Generates animations of the clustering process
- Testing capabilities for KMeans against the version from Scikit-learn

## Animation

Evolution of the DBSCAN algorithm run on the smile data set.

![til](./dbscan_animation.gif)

## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
