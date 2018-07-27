import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

class Graph(object):
    """
    A flexible graphing utility
    """
    def __init__(self,title,data,grid=True):
        self.title = title
        self.fig = plt.figure(figsize=(5,5))
        self.fig.canvas.set_window_title(self.title)
        self.ax = self.fig.add_subplot(111, projection='3d')
        if grid == True:
            if len(data.shape) > 3:
                print("Cannot plot more than 4 dimensions")
                self.data = np.array([[],[],[]])
            else:
                self.data = data
            if self.data.shape[1] == 2:
                self.ax.scatter(self.data[:,0],self.data[:,1],0)
            elif self.data.shape[1] == 3:
                self.ax.scatter(self.data[:,0],self.data[:,1],self.data[:,2], zdir='z', cmap='plasma',c=-self.data[:,1])
        else:
            z,x,y = np.squeeze(data).nonzero()
            self.ax.set_xlim([0,data.shape[0]])
            self.ax.set_ylim([0,data.shape[1]])
            self.ax.set_zlim([0,data.shape[2]])
            self.ax.scatter(x, y, z, zdir='z', cmap='plasma',c=-y)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        plt.axis('scaled')

    def show(self):
        plt.show()
