import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import chi2
from scipy.special import erf
import plotly.graph_objects as go
import imageio
import os
import shutil

class GmmViz:
    def __init__(self, gmm, colors:list = None , utiPlotly = False):
        self.gmm = gmm
        self.data = gmm.data
        self.dim = gmm.data.shape[1]
        self.k = gmm.n_clusters
        self.mean = gmm.getEstimands('mean')
        self.Sigma = gmm.getEstimands('Sigma')
        self.LL = gmm.getEstimands('log_likelihood')
        self.n_iter = gmm.n_iter_
        
        if colors is not None:
            # confirm the number of colors is equal to the number of clusters
            if len(colors) != self.k:
                raise ValueError("The number of colors must be equal to the number of clusters.")
            self.colors = colors
        else:
            palette = ['#780000', '#03045e', '#f48c06', '#6d8484', '#88976d', '#5e5e9b', '#a64d4d', '#814da8', '#775a5a'] #len: 9
            if self.dim > len(palette):
                raise ValueError("The number of clusters is greater than the number of colors available by default, pass the list of color to 'colors'.")
            self.colors = palette[:self.k]
            
        self.useplotly = utiPlotly

    def _plot_gaussian_2d(self, mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
        '''
        Utility function to plot one Gaussian from mean and covariance.
        '''
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
                          width=ell_radius_x * 2,
                          height=ell_radius_y * 2,
                          facecolor=facecolor,
                          **kwargs)
        
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = mean[0]
        
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = mean[1]
        
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        
        ellipse.set_transform(transf + ax.transData)
        
        return ax.add_patch(ellipse)
    
    def _eigen_decomp(self, cov):
        '''
        Function to perform eigenvalue decomposition of the covariance matrix.
        '''
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1] # Sort by descending eigenvalues
        return eigenvalues[order], eigenvectors[:, order]
    
    def _plot_gaussian_3d(self, mean, cov, ax, color, n_std=2, alpha = 0.4, **kwargs):
        if self.useplotly:
            # Calculate the eigenvalues and eigenvectors for the covariance matrix
            eigenvalues, eigenvectors = self._eigen_decomp(cov = cov)
            
            # Adjust radii by the chi-squared distribution quantile for confidence interval
            radii = np.sqrt(eigenvalues) * n_std * np.sqrt(chi2.ppf((1 + erf(n_std / np.sqrt(2))) / 2, df=2))

            # Generate a grid of points representing the ellipsoid
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = radii[0] * np.outer(np.sin(v), np.cos(u))
            y = radii[1] * np.outer(np.sin(v), np.sin(u))
            z = radii[2] * np.outer(np.cos(v), np.ones_like(u))

            # Rotate and translate the points according to the eigenvectors
            # Rotate and translate points
            x, y, z = x.flatten(), y.flatten(), z.flatten()
            points = np.vstack((x, y, z)).T @ eigenvectors + mean
            x, y, z = points[:, 0].reshape((100, 100)), points[:, 1].reshape((100, 100)), points[:, 2].reshape((100, 100))

            # Using a single color with the correct format
            color_value = np.full(x.shape, color)  # Example value; adjust as necessary for proper coloring
            ax.add_trace(go.Surface(x=x, y=y, z=z, opacity=alpha, surfacecolor=color_value, colorscale=[[0, color], [1, color]], showscale=False))
            
        else:
            ## Calculate the eigenvalues and eigenvectors for the covariance matrix
            eigenvalues, eigenvectors = self._eigen_decomp(cov = cov)
            
            # Adjust radii by the chi-squared distribution quantile for confidence interval
            radii = np.sqrt(eigenvalues) * n_std * np.sqrt(chi2.ppf((1 + erf(n_std / np.sqrt(2))) / 2, df=2))

            # Generate a grid of points representing the ellipsoid
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = radii[0] * np.outer(np.sin(v), np.cos(u))
            y = radii[1] * np.outer(np.sin(v), np.sin(u))
            z = radii[2] * np.outer(np.cos(v), np.ones_like(u))

            # Rotate the points according to the eigenvectors
            ellipsoid = np.dstack((x, y, z))
            ellipsoid = ellipsoid @ eigenvectors + mean

            # Plot the surface
            ax.plot_surface(ellipsoid[:,:,0], ellipsoid[:,:,1], ellipsoid[:,:,2], color = color, alpha=alpha)
    
    def _draw(self, ax, mean, Sigma, n_std=2.0, **kwargs):
        '''
        Function to draw the Gaussians.
        Note: Only for two-dimensionl dataset
        '''
        
        if self.dim == 2:
            for i in range(self.k):
                self._plot_gaussian_2d(np.array(mean)[i], np.array(Sigma)[i], ax, n_std = n_std, edgecolor = self.colors[i], **kwargs)
        elif self.dim == 3 :
            if self.useplotly:
                for i in range(self.k):
                    self._plot_gaussian_3d(np.array(mean)[i], np.array(Sigma)[i], ax, color = self.colors[i], n_std = n_std)
            else:
                for i in range(self.k):
                    self._plot_gaussian_3d(np.array(mean)[i], np.array(Sigma)[i], ax, color = self.colors[i], n_std = n_std, edgecolor = self.colors[i], **kwargs)

    def _plot_iter(self, X, mean, Sigma, title, filename, show_plot):
        if self.useplotly:
            if self.dim != 3:
                raise ValueError("Only 3D data can be visualized with Plotly in this setting.")
            fig = go.Figure()
            
            # Scatter plot for data points
            fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers', marker=dict(color='black', size=3, opacity=0.5)))
            
            # Scatter plot for centroids
            fig.add_trace(go.Scatter3d(
                x=np.array(mean)[:, 0], y=np.array(mean)[:, 1], z=np.array(mean)[:, 2],
                mode='markers',  # Add 'text' to mode if you want labels
                marker=dict(
                    color = self.colors,  # Bright and distinct colors
                    size = 8,  # Larger size
                    opacity = 1  # No transparency
                )
            ))

            self._draw(fig, mean, Sigma)
            
            # Update plot layout
            fig.update_layout(
                title=title,
                autosize=True,
                scene=dict(
                    xaxis=dict(range=[np.min(X[:, 0])-1, np.max(X[:, 0])+1]),
                    yaxis=dict(range=[np.min(X[:, 1])-1, np.max(X[:, 1])+1]),
                    zaxis=dict(range=[np.min(X[:, 2])-1, np.max(X[:, 2])+1])
                )
            )
            
            # show plots
            if show_plot:
                fig.show()
                
            # Save plots
            fig.write_image(filename)
                
        else:
            if self.dim == 2:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(X[:, 0], X[:, 1], s=3, alpha=0.5, c='black')
                # Scatter plot for centroids with the corresponding colors
                ax.scatter(np.array(mean)[:, 0], np.array(mean)[:, 1], c=self.colors)

            elif self.dim == 3:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                # Scatter plot for data points
                ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=3, alpha=0.5, c='black')
                # Scatter plot for centroids with the corresponding colors
                ax.scatter(np.array(mean)[:, 0], np.array(mean)[:, 1], np.array(mean)[:, 2], c=self.colors)

            else:
                raise ValueError("Only 2D and 3D data can be visualized.")

           
            self._draw(ax, mean, Sigma, lw=3)
            
            ax.set_xlim(min(X[:, 0]), max(X[:, 0]))
            ax.set_ylim(min(X[:, 1]), max(X[:, 1]))
            if self.dim == 3:
                ax.set_zlim(min(X[:, 2]), max(X[:, 2]))

            plt.title(title)
            plt.savefig(filename, dpi = 200)
            if show_plot:
                plt.show()
            plt.clf()
        
    def plot(self, fig_title = "", path_prefix = "", X = None, max_iter = 15, show_plot = False):
        '''
        Draw the data points and the fitted mixture model.
        input:
            - title: title of plot and name with which it will be saved.
        '''
        if X is  None:
            X = self.data
        
        self.clean_directory(path_prefix)
        
        if (max_iter is None) or (max_iter > self.n_iter):
            for i in range(self.n_iter):
                iter_title = f"{fig_title} Iteration {i:02d}"
                filename = f"{path_prefix}Iter_{i:02d}.png"
                self._plot_iter(X, self.mean[i], self.Sigma[i], iter_title, filename, show_plot = show_plot)
        else:
            for i in range(max_iter):
                iter_title = f"{fig_title} Iteration {i:02d}"
                filename = f"{path_prefix}Iter_{i:02d}.png"
                self._plot_iter(X, self.mean[i], self.Sigma[i], iter_title, filename, show_plot = show_plot)

    def plot_likelihood(self, output_path_filename, dpi = 300):
        
        self.clean_directory(output_path_filename)
        
        for i in range(1, len(self.LL)):
            plt.title("log-likelihood for iteration: " + str(i))
            plt.plot(self.LL[1:1+i], marker='.')
            axes = plt.gca()
            axes.set_ylim([min(self.LL[1:])-200, max(self.LL[1:])+200]) 
            axes.set_xlim([-1, len(self.LL)])
            if i <= 9 :
                I = "0"+str(i)
                plt.savefig(output_path_filename + "Log-likelihood" + str(I) + ".png", dpi = dpi)
            else:   
                plt.savefig(output_path_filename + "Log-likelihood" + str(i) + ".png", dpi = dpi)
            plt.clf()
    
    @staticmethod
    def generateGIF(image_path, output_path_filename, fps = 2):

        folder_path = image_path
        filenames = sorted([img for img in os.listdir(folder_path) if img.endswith('.png')])

        # Create GIF
        with imageio.get_writer(output_path_filename, mode='I', fps = fps) as writer:
            for filename in filenames:
                image_path = os.path.join(folder_path, filename)
                image = imageio.imread(image_path)
                writer.append_data(image)

        print(f"GIF saved as {output_path_filename.split('/')[-1]} in {os.path.dirname(output_path_filename)}")
    
    @staticmethod    
    def clean_directory(dir_path):
        # Check if the directory exists
        if not os.path.exists(dir_path):
            print("Directory does not exist.")
            return

        # Loop through all files in the directory
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Removes file or link
                elif os.path.isdir(file_path):
                    continue
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')