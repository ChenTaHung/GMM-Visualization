#%%
import numpy as np
import os 
# os.chdir("/Users/chentahung/Desktop/MSSP/MA589-ComputationalStatistics/FinalProject/")
os.chdir("/Users/shenfengyuan/Desktop/BU-MSSP/MA589/ma589-proj-final-gentlemen_in_massachusetts")

# comment my path and add urs

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from src.main.GMMViz.GaussianMixtureModel import GMM
from main.GMMViz.GmmPlot import Viz
import plotly.io as pio
# %%
# 2D Gaussian Mixture Model

"""
2 DIMENSIONAL GAUSSIAN MIXTURE MODEL
"""

np.random.seed(128)
X2 = Viz.genData(k=3, dim=2, points_per_cluster=200, lim=[-10, 10], plot = True)
gmm2 = GMM(dataset=X2, n_clusters=3)
gmm2.fit()
# %%
V2 = Viz(gmm2, colors=['#D90429', '#0096c7', '#FFD60A'])
V2.plot(X2, title="")
Viz.generateGIF(image_path = "doc/image/dim2/parms", output_path_filename = "doc/image/dim2/parms/gif/GMM-2D-Parms.gif", fps = 2)
#%%
V2.plot_likelihood(output_path_filename="doc/image/dim2/ll/")
Viz.generateGIF(image_path = "doc/image/dim2/ll", output_path_filename = "doc/image/dim2/ll/gif/GMM-2D-LL.gif", fps = 2)

# %%
"""
3 DIMENSIONAL GAUSSIAN MIXTURE MODEL
"""

# 3D Gaussian Mixture Model
X3 = Viz.genData(k=3, dim=3, points_per_cluster=200, lim=[-10, 10], plot = True, random_state = 129)
gmm3 = GMM(dataset=X3, n_clusters=3, random_state=129, max_iter = 30)
gmm3.fit()

#%%
pio.renderers.default = "png"
color_list = plt.cm.Set1(np.linspace(0, 1, 3))
V3F = Viz(gmm3, colors=color_list[:, :3], utiPlotly=False)
V3F.plot(X3, "doc/image/dim3/parms/Iter:")
Viz.generateGIF(image_path = "doc/image/dim3/parms", output_path_filename = "doc/image/dim3/parms/gif/GMM-3D-Parms.gif", fps = 2)

#%%
V3F.plot_likelihood(output_path_filename="doc/image/dim3/ll/")
Viz.generateGIF(image_path = "doc/image/dim3/ll", output_path_filename = "doc/image/dim3/ll/gif/GMM-3D-LL.gif", fps = 2)
# %%
"""
Interactive 3D plot
"""
pio.renderers.default = "browser"

V3T = Viz(gmm3, utiPlotly=True)
V3T.plot(X3, "GMM-3D")
# %%
