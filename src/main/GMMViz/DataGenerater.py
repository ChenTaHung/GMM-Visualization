import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class DataGenerater:
    @staticmethod
    def genData(k=3, dim=2, points_per_cluster=200, lim=[-10, 10], scale = 5, plot = True, useplotly = True, random_state = None):
        '''
        Generates data from a random mixture of Gaussians in a given range.
        Will also plot the points in case of 2D.
        input:
            - k: Number of Gaussian clusters
            - dim: Dimension of generated points
            - points_per_cluster: Number of points to be generated for each cluster
            - lim: Range of mean values
            - scale: Scaling factor for covariance matrix
        output:
            - X: Generated points (points_per_cluster*k, dim)
        '''
        if dim > 3 : # no visualization for dim > 3
            plot = False
        
        if random_state:
            random.seed(random_state)
        
        x = []
        
        mean = random.rand(k, dim)*(lim[1]-lim[0]) + lim[0]
        
        for i in range(k):
            cov = random.rand(dim, dim * scale)
            cov = np.matmul(cov, cov.T) #matrix product
            _x = np.random.multivariate_normal(mean[i], cov, points_per_cluster)
            x += list(_x)
        x = np.array(x)
        if plot:
            if (dim == 2):
                fig = plt.figure()
                ax = fig.gca()
                ax.scatter(x[:,0], x[:,1], s=3, alpha=0.4)
                ax.autoscale(enable=True) 
            elif (dim == 3):
                if useplotly:
                    fig = go.Figure(data=[go.Scatter3d(
                        x=x[:, 0],  # X data
                        y=x[:, 1],  # Y data
                        z=x[:, 2],  # Z data
                        mode='markers',
                        marker=dict(
                            size=3,         # Size of markers
                            opacity=0.4     # Opacity of markers
                        )
                    )])

                    # Updating layout (correcting the auto range setting)
                    fig.update_layout(
                        scene=dict(
                            xaxis=dict(autorange=True),  # Correct property for auto-ranging
                            yaxis=dict(autorange=True),  # Correct property for auto-ranging
                            zaxis=dict(autorange=True)   # Correct property for auto-ranging
                        ),
                        width=700,  # Width of the figure (optional)
                        height=700  # Height of the figure (optional)
                    )

                    fig.show()
                else:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(x[:,0], x[:,1], x[:,2], s=3, alpha=0.4)
                    ax.autoscale(enable=True)
            else:
                raise ValueError("Only 2D and 3D data can be visualized.")
        return x