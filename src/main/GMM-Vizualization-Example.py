#%%
import numpy as np
import os
os.chdir("/Users/chentahung/Desktop/git/GMM-Visualization")

from src.main.GMMViz.GaussianMixtureModel import GMM
from src.main.GMMViz.GmmPlot import GmmViz
from src.main.GMMViz.DataGenerater import DataGenerater

import plotly.io as pio
# %%
# 2D Gaussian Mixture Model

"""
2 DIMENSIONAL GAUSSIAN MIXTURE MODEL
"""

np.random.seed(128)
X2 = DataGenerater.genData(k=3, dim=2, points_per_cluster=200, lim=[-10, 10], plot = True)
gmm2 = GMM(n_clusters=3, random_state=129)
gmm2.fit(X2)
# %%

V2 = GmmViz(gmm2)

# plot convergence
V2.plot(fig_title="GMM-2D", path_prefix="doc/image/dim2/parms/")

# generate gif ( need to plot the convergence first)
GmmViz.generateGIF(image_path = "doc/image/dim2/parms", output_path_filename = "doc/image/dim2/parms/gif/GMM-2D-Parms.gif", fps = 2)

#%%
# Likelihood
V2.plot_likelihood(output_path_filename="doc/image/dim2/ll/")
GmmViz.generateGIF(image_path = "doc/image/dim2/ll", output_path_filename = "doc/image/dim2/ll/gif/GMM-2D-LL.gif", fps = 2)

# %%
"""
3 DIMENSIONAL GAUSSIAN MIXTURE MODEL
"""

# 3D Gaussian Mixture Model
pio.renderers.default = "notebook" 
X3 = DataGenerater.genData(k=3, dim=3, points_per_cluster=200, lim=[-10, 10], plot = True, random_state = 129)
gmm3 = GMM(n_clusters=3)
gmm3.fit(X3)

#%%
pio.renderers.default = "notebook"  # show in interactive editor (ipython notebook)
                                                                                    
V3F = GmmViz(gmm3, utiPlotly=False)
V3F.plot(fig_title="GMM-3D", path_prefix="doc/image/dim3/parms/", show_plot=False)
GmmViz.generateGIF(image_path = "doc/image/dim3/parms", output_path_filename = "doc/image/dim3/parms/gif/GMM-3D-Parms.gif", fps = 2)

#%%
V3F.plot_likelihood(output_path_filename="doc/image/dim3/ll/")
GmmViz.generateGIF(image_path = "doc/image/dim3/ll", output_path_filename = "doc/image/dim3/ll/gif/GMM-3D-LL.gif", fps = 2)
# %%
"""
3 DIMENSIONAL GAUSSIAN MIXTURE MODEL
    +
Interactive 3D plot
"""
pio.renderers.default = "png"

V3T = GmmViz(gmm3, utiPlotly=True)
V3T.plot(fig_title = "GMM-3D", path_prefix="doc/image/dim3/parms/", show_plot = False)

#%%
"""
Over 3 dimensions : Using PCA 
    +
Interactive 3D plot
"""
X7 = DataGenerater.genData(k=6, dim=7, points_per_cluster=100, lim=[-20, 20], plot = False, random_state = 129)
PCAGMM = GMM(n_clusters=6)

PCAGMM.PCA_fit(X = X7, n_components=3)
#%%
# plot
pio.renderers.default = "browser"

V7T = GmmViz(PCAGMM, utiPlotly=True)
V7T.plot(fig_title = "GMM-3D")
# %%
