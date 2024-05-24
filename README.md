<h1 align="center"><b>GMM-Visualization</b></h1>


The project is a visualization tool kit to visualize how a `Gaussian Mixture Model` converges, and interactively show it in a 3d space. Although the project focus on visualizing in 3D spaces, it also support plots in 2D.

<h2><b>Installation</b></h2>

First, clone the GitHub repository

```bash
git clone https://github.com/ChenTaHung/GMM-Visualization path/you/want/to/clone
git clone git@github.com:ChenTaHung/GMM-Visualization.git path/you/want/to/clone
```
Then, switch to the directory where the repository has been cloned.

```python
import numpy as np
import os
os.chdir('/path/to/the/cloned/repository')
from src.main.GMMViz.GaussianMixtureModel import GMM
from src.main.GMMViz.GmmPlot import GmmViz
from src.main.GMMViz.DataGenerater import DataGenerater

import plotly.io as pio
```

<h2><b>Usage</b></h2>

<h3> Quick example: </h3>

#### 1. Change directory to the path you cloned the repository to.

```python
import numpy as np
import os 
os.chdir("/folder/that/you/cloned/")

from src.main.GMMViz.GaussianMixtureModel import GMM
from src.main.GMMViz.GmmPlot import GmmViz
from src.main.GMMViz.DataGenerater import DataGenerater
import plotly.io as pio
```

#### 2. Generating test case dataset. (Or load your own dataset)

#### 2.1 Working with data under 3 dimensions.

```python
"""
3D Gaussian Mixture Model
"""

pio.renderers.default = "notebook"

# Generate dataset with k = 3 groups within a dim = 3 dimensional space. 
X3 = DataGenerater.genData(k = 3,  # used to generate data with clearly k clusters.
                           dim = 3, # dimension of the data
                           points_per_cluster = 200, 
                           lim = [-10, 10], # range of mean values for each clusters
                           plot = True, # only data with dimension lower than 3 can be plotted.
                           random_state = 129)
```

<p align = 'center'><img src = 'https://github.com/ChenTaHung/GMM-Visualization/blob/main/doc/image/data/k3d3.png' alt = 'Image' style = 'width: 800px'/></p>

```python
# instantiate the object
gmm3 = GMM(n_clusters=3, random_state=129)
# fit the GMM to the data
gmm3.fit(X3)
```

`X3` can be replaced by a pandas dataframe or a numpy array.

You can use `gmm3.getEstimands(parm = )` with arguments options: `['mean', 'Sigma', 'log_likelihood']`, to get the corrsponding parameter information in the covergence of the GMM. If no argument passed, then it will return the dictionary of the parameters estimation in a dictionary.

#### 2.2 Working with data under 3 dimensions.

When a dataset exceeds 3 dimensions, visualizing it directly in 3D space is impractical. `Principal Component Analysis (PCA)` addresses this by reducing the dataset's dimensionality. It projects the data onto the top three directions of maximum variance, identified through eigenvectors of the covariance matrix. Setting the number of principal components (`n_component`) to **3** allows the transformed dataset to be visualized effectively in three-dimensional space.

```python
"""
Over 3 dimensions : Using PCA 
"""
X7 = DataGenerater.genData(k=6, dim=7, points_per_cluster=100, lim=[-20, 20], plot = False, random_state = 129) # plot = False, since the data with greater than 3 dimensions is not able to visualized.
PCAGMM = GMM(n_clusters=6)

PCAGMM.PCA_fit(X = X7, n_components=3) # n_components' default value is 3, which is to form a 3 dimensional data.
```

#### 3. Plot the Gaussian distribution.

There are two options for plotting the GMM in 3 dimensional space.

The `plot()` method draw the multivariate Gaussian distribution as a ellipsoid for each cluster.

#### 3.1 Using matplotlib.pyplot (set `utiPlotly = False`):

```python
# instantiate the GmmViz object
V3F = GmmViz(gmm3, utiPlotly=False) # plot via matplotlib

# use plot method to plot
V3F.plot(fig_title="GMM-3D", 
         path_prefix="doc/image/dim3/parms/", # image will be stored in the `path_prefix` directory.
         show_plot = False, #  tells whether to show the figure through the editor or not. Default is `False`.
         max_iter = 15) # number of iteration to plot. Default is 15.
```

In `plot()` method, the `show_plot` parameter tells whether to show the figure through the editor or not. Default is `False`.

We can genreate gif file from the images we exported by the `plot()` method.

```python
GmmViz.generateGIF(image_path = "doc/image/dim3/parms", # directory of the images showing each iteraction
                   output_path_filename = "doc/image/dim3/parms/gif/GMM-3D-Parms.gif", 
                   fps = 2) # Adjust the timing of each frame in the GIF file

```

<p align = 'center'><img src = 'https://github.com/ChenTaHung/GMM-Visualization/blob/main/doc/image/dim3/parms/gif/GMM-3D-Parms.gif' alt = 'Image' style = 'width: 800px'/></p>

#### 3.2 Using Plotly (set `utiPlotly = True`):

```python
`"""
Interactive 3D plot
"""
# plot
pio.renderers.default = "browser" # it will open the browser to show the plots.

# GMM for 3 dim dataset
V3T = GmmViz(gmm3, utiPlotly=True)
V3T.plot(fig_title = "GMM-3D", path_prefix="doc/image/dim3/parms/", show_plot = False) # the parameters doesn't effect if you show in browser.

# PCA_fit GMM
V7T = GmmViz(PCAGMM, utiPlotly=True)
V7T.plot(fig_title = "GMM-3D")
```

#### 4. Visualize in 2D spaces.

```python

"""
2 DIMENSIONAL GAUSSIAN MIXTURE MODEL
"""

np.random.seed(128)
X2 = DataGenerater.genData(k=3, dim=2, points_per_cluster=200, lim=[-10, 10], plot = True)
gmm2 = GMM(n_clusters=3, random_state=129)
gmm2.fit(X2)

V2 = GmmViz(gmm2)

# plot convergence
V2.plot(fig_title="GMM-2D", path_prefix="doc/image/dim2/parms/")

# generate gif ( need to plot the convergence first)
GmmViz.generateGIF(image_path = "doc/image/dim2/parms", output_path_filename = "doc/image/dim2/parms/gif/GMM-2D-Parms.gif", fps = 2)

# Likelihood
V2.plot_likelihood(output_path_filename="doc/image/dim2/ll/")
GmmViz.generateGIF(image_path = "doc/image/dim2/ll", output_path_filename = "doc/image/dim2/ll/gif/GMM-2D-LL.gif", fps = 2)
```

<p align = 'center'><img src = 'https://github.com/ChenTaHung/GMM-Visualization/blob/main/doc/image/dim2/parms/gif/GMM-2D-Parms.gif' alt = 'Image' style = 'width: 800px'/></p>


<h2><b>Environment</b></h2>

```bash
OS : macOS Sonoma 14.5
IDE: Visual Studio Code 
Language : Python       3.9.7 

Package list:
backports.shutil-get-terminal-size 1.0.0
imageio                            2.9.0
matplotlib                         3.7.2
matplotlib-inline                  0.1.6
numpy                              1.20.3
numpydoc                           1.1.0
pandas                             1.5.3
plotly                             5.21.0
scipy                              1.10.1
```

<h2><p><b>Developers</b></p></h2>

Denny Chen
   -  LinkedIn Profile : https://www.linkedin.com/in/dennychen-tahung/
   -  E-Mail : denny20700@gmail.com
