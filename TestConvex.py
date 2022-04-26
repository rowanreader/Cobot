import numpy as np
import pyvista as pv
import cdd as pcdd
from scipy.spatial import ConvexHull

tile = [(19, 70), (83, 19), (35, -55), (10, -43), (-51, -46), (-34, 0), (-52, 35), (62, 15)]
pillars = [np.array([483., 110.]), np.array([499., 122.]), np.array([520., 110.])]
origin = [[508.75, 113.25]]

numT = len(tile)
numP = len(pillars)

# gotta add z component
newTile = []
newPillars = []
for i in tile:
    newTile.append([i[0] + origin[0][0], i[1] + origin[0][1], 0])
for i in pillars:
    newPillars.append(np.append(i, 0))

# take one cube
cube1 = pv.Cube()
# take the same cube but translate it
cube2 = pv.Cube()
cube2.translate((0.5, 0.5, 0.5))

# plot
# pltr = pv.Plotter(window_size=[512,512])
# pltr.add_mesh(cube1)
# pltr.add_mesh(cube2)
# pltr.show()

# I don't know why, but there are duplicates in the PyVista cubes;
# here are the vertices of each cube, without duplicates
pts1 = cube1.points[0:8, :]
pts2 = cube2.points[0:8, :]

# make the V-representation of the first cube; you have to prepend
# with a column of ones
v1 = np.column_stack((np.ones(numT), newTile))
# v1 = np.column_stack((np.ones(8), pts1))
mat = pcdd.Matrix(v1, number_type='fraction') # use fractions if possible
mat.rep_type = pcdd.RepType.GENERATOR
poly1 = pcdd.Polyhedron(mat)

# make the V-representation of the second cube; you have to prepend
# with a column of ones
v2 = np.column_stack((np.ones(numP), newPillars))
# v2 = np.column_stack((np.ones(8), pts2))
mat = pcdd.Matrix(v2, number_type='fraction')
mat.rep_type = pcdd.RepType.GENERATOR
poly2 = pcdd.Polyhedron(mat)

# H-representation of the first cube
h1 = poly1.get_inequalities()

# H-representation of the second cube
h2 = poly2.get_inequalities()

# join the two sets of linear inequalities; this will give the intersection
hintersection = np.vstack((h1, h2))

# make the V-representation of the intersection
mat = pcdd.Matrix(hintersection, number_type='fraction')
mat.rep_type = pcdd.RepType.INEQUALITY
polyintersection = pcdd.Polyhedron(mat)

# get the vertices; they are given in a matrix prepended by a column of ones
vintersection = polyintersection.get_generators()

# get rid of the column of ones
num = len(vintersection)
ptsintersection = np.array([
    vintersection[i][1:4] for i in range(num)
])

# these are the vertices of the intersection; it remains to take
# the convex hull
hull = ConvexHull(ptsintersection)

print(hull.area)