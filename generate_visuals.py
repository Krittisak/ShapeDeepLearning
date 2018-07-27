import numpy as np
from bin.shapes import Sphere, Torus, Cube, SquareTorus, Pyramid, Gridspace
#from graph import Graph
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# template = Torus(classarray=np.array([1,0]))
# #template = Sphere(classarray=np.array([1,0]))
gridspace = Gridspace(stepsize=.5, radius=20)
# default = {"Rotation":0,
#            "Scaling":0,
#            "Noise":0,
#            "Translation":0,
#            "Linear Warp":0,
#            "Poly Warp":0}
# sampling = {"High":10000,"Medium":1000,"Low":100}
# settings1 = ["Default", "Scaling", "Rotation", "Translation", "Linear Warp", "Poly Warp", "Noise"]
# settings2 = ["High", "Medium", "Low"]
# fig = plt.figure(figsize=(16,8))
# fig.canvas.set_window_title("Shape Exemplars")
# k = 1
# axes = {}
# for i, setting2 in enumerate(settings2):
#     for j, setting1 in enumerate(settings1):
#         axes[setting2+setting1] = fig.add_subplot(len(settings2),len(settings1),k, projection='3d')
#         axes[setting2+setting1].set_title(setting1+", Sampling "+setting2, fontsize=8)
#
#         shapeinfo = {"scale":7,
#                      "rotation":default["Rotation"],
#                      "scale_randomness":default["Scaling"],
#                      "noise":default["Noise"],
#                      "translation":default["Translation"],
#                      "transformation":default["Linear Warp"],
#                      "polyscale":default["Poly Warp"]}
#         if setting1 == "Rotation":
#             shapeinfo['rotation']=1
#         if setting1 == "Scaling":
#             shapeinfo['scale_randomness']=3
#         if setting1 == "Noise":
#             shapeinfo['noise']=1
#         if setting1 == "Translation":
#             shapeinfo['translation']=4
#         if setting1 == "Linear Warp":
#             shapeinfo['transformation']=.3
#         if setting1 == "Poly Warp":
#             shapeinfo['polyscale']=.5
#         template.set_transforms(**shapeinfo)
#         points = template.sample(sampling[setting2])
#         x,y,z = template.as_grid(gridspace).nonzero()
#         axes[setting2+setting1].set_xlim([0,gridspace.shape[0]])
#         axes[setting2+setting1].set_ylim([0,gridspace.shape[1]])
#         axes[setting2+setting1].set_zlim([0,gridspace.shape[2]])
#         axes[setting2+setting1].scatter(x, y, z, zdir='z', cmap='plasma',c=-y)
#         #ax.set_xlabel("X")
#         #ax.set_ylabel("Y")
#         #ax.set_zlabel("Z")
#         #del shapeinfo
#         k += 1
# plt.tight_layout()
#plt.savefig('./writeup/graphics/shape_exemplars_to.eps')
#plt.savefig('./writeup/graphics/shape_exemplars_to.pdf')

s = Sphere(classarray=np.array([1,0]))
t = Torus(classarray=np.array([1,0]))
c = Cube(classarray=np.array([1,0]))
q = SquareTorus(classarray=np.array([1,0]))
p = Pyramid(classarray=np.array([1,0]))

s.set_transforms(scale=15)
t.set_transforms(scale=15)
c.set_transforms(scale=15)
q.set_transforms(scale=15)
p.set_transforms(scale=15)

s.sample(10000)
t.sample(10000)
c.sample(10000)
q.sample(10000)
p.sample(10000)

fig = plt.figure(figsize=(15,3))
fig.canvas.set_window_title("Shapes Library")

axs = fig.add_subplot(1,5,1, projection='3d')
axs.set_title("Sphere", fontsize=10)
x,y,z = s.as_grid(gridspace).nonzero()
axs.set_xlim([0,gridspace.shape[0]])
axs.set_ylim([0,gridspace.shape[1]])
axs.set_zlim([0,gridspace.shape[2]])
axs.scatter(x, y, z, zdir='z', cmap='plasma',c=-y)

axt = fig.add_subplot(1,5,2, projection='3d')
axt.set_title("Torus", fontsize=10)
x,y,z = t.as_grid(gridspace).nonzero()
axt.set_xlim([0,gridspace.shape[0]])
axt.set_ylim([0,gridspace.shape[1]])
axt.set_zlim([0,gridspace.shape[2]])
axt.scatter(x, y, z, zdir='z', cmap='plasma',c=-y)

axc = fig.add_subplot(1,5,3, projection='3d')
axc.set_title("Cube", fontsize=10)
x,y,z = c.as_grid(gridspace).nonzero()
axc.set_xlim([0,gridspace.shape[0]])
axc.set_ylim([0,gridspace.shape[1]])
axc.set_zlim([0,gridspace.shape[2]])
axc.scatter(x, y, z, zdir='z', cmap='plasma',c=-y)

axq = fig.add_subplot(1,5,4, projection='3d')
axq.set_title("Square Torus", fontsize=10)
x,y,z = q.as_grid(gridspace).nonzero()
axq.set_xlim([0,gridspace.shape[0]])
axq.set_ylim([0,gridspace.shape[1]])
axq.set_zlim([0,gridspace.shape[2]])
axq.scatter(x, y, z, zdir='z', cmap='plasma',c=-y)

axp = fig.add_subplot(1,5,5, projection='3d')
axp.set_title("Pyramid", fontsize=10)
x,y,z = p.as_grid(gridspace).nonzero()
axp.set_xlim([0,gridspace.shape[0]])
axp.set_ylim([0,gridspace.shape[1]])
axp.set_zlim([0,gridspace.shape[2]])
axp.scatter(x, y, z, zdir='z', cmap='plasma',c=-y)
plt.tight_layout()
plt.savefig('./writeup/graphics/shape_library.eps')
plt.savefig('./writeup/graphics/shape_library.pdf')
