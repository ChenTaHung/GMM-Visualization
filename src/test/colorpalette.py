import matplotlib.pyplot as plt

# List of colors
colors = ['#780000', '#03045e', '#f48c06', '#6d8484', '#88976d', '#5e5e9b', '#a64d4d', '#814da8', '#775a5a']

# Setting up the plot
fig, ax = plt.subplots()

# Plotting each color as a dot on the plot
for i, color in enumerate(colors):
    ax.plot(i, 1, 'o', markersize=20, color=color)

# Setting plot parameters to make it cleaner
ax.set_xlim(-1, len(colors))
ax.set_ylim(0.5, 1.5)
ax.axis('off')

plt.show()
