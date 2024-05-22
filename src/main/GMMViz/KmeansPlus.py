import numpy as np

class Kmeans_Plus:
    
    def __init__(self, dataset, k) :
        self.dataset = dataset
        self.k = k
        self.n = dataset.shape[1]
        self.m = dataset.shape[0]
    
    def _calculateEucDistance(self, vec1, vec2):
    
        return np.sqrt(np.sum(np.power(vec1-vec2, 2)))

    def _get_closest_dist(self, point, centroids):
        """
        Calculates the closest distance between a given point and a list of centroids.

        Parameters:
        - point: A numpy array representing the coordinates of the point.
        - centroids: A list of numpy arrays representing the coordinates of the centroids.

        Returns:
        - min_dist: The minimum distance between the point and the centroids.
        """
        min_dist = np.inf
        for centroid in centroids:
            dist = self._calculateEucDistance(np.array(point), np.array(centroid))
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _rouletteWheelSelect(self, P, r):
        '''
        P: prob array
        r: random number
        '''
        q = 0
        for i in range(len(P)):
            q += P[i]
            if i == (len(P)-1):
                q = 1
            if r <= q :
                return i

    def getCenter(self, random_state = None):
        """
        Returns the centroids and the index of the chosen centroid for each cluster.

        Returns:
            centroids (numpy.matrix): The centroids of each cluster.
            choosen_index (int): The index of the chosen centroid for each cluster.
        """
        centroids = np.mat(np.zeros((self.k, self.n)))
        centroids_index = np.zeros(self.k)
        if random_state:
            np.random.seed(random_state)
        index = np.random.randint(0, self.n)

        centroids[0] = self.dataset[index,:]

        d = np.mat(np.zeros((self.m, 1)))

        for i in range(1, self.k):
            for j in range(self.m):
                d[j, 0] = self._get_closest_dist(self.dataset[j,:], centroids)

            P = np.square(d) / np.square(d).sum()
            r = np.random.random()
            choosen_index = self._rouletteWheelSelect(P, r)
            centroids[i, :] = self.dataset[choosen_index,:]
            centroids_index[i] = choosen_index

        return centroids, centroids_index