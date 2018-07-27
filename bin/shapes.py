import numpy as np
import math
import random
import scipy.misc
from numpy.polynomial import Polynomial as P
from numpy.polynomial.polynomial import polyval2d as p2d
from scipy.linalg import qr
from scipy.stats import ortho_group
try:
    from poly import applyfixedpolynomial
except:
    from bin.poly import applyfixedpolynomial



class Gridspace(object):
    """
    A utility class for specifying a grid upon which a shape can be projected
    """

    def __init__(self, stepsize=.1, radius=10):
        self.stepsize = stepsize
        self.center = np.array([0, 0, 0])
        self.radius = radius
        self.shape = list(np.full(3, int(self.radius*2/self.stepsize)))

    def get_index(self, point):
        for i, p in enumerate(point):
            if self.center[i]-self.radius <= p <= self.center[i]+self.radius+self.stepsize:
                pass
            else:
                #print("break")
                return None
        actual = np.floor((point+self.radius)/self.stepsize).astype(int)
        for i, p in enumerate(actual):
            if 0 <= p <= self.shape[i]:
                return tuple(actual)
            else:
                #print("caught")
                return None

    def get_empty(self):
        return np.zeros(self.shape)


class Shape(object):
    """
    A generic template for shapes
    """

    def __init__(self, classarray):
        self.noise = 0
        self.scale = 1
        self.scale_randomness = 0
        self.translation = 0
        self.transformation = 0
        self.rotation = 0
        self.classarray = classarray

        self.polyvars = 1
        self.polydegree = 1
        self.polyscale = 1
        self.polycoef = []
        self.polyrandom = -1

    def sample(self, n):
        return []

    def transform(self, array):
        if self.polyscale != 0:
            if self.polyrandom == 0:
                array = applyfixedpolynomial(array, self.polyvars, self.polycoef, self.polyscale)
            if self.polyrandom == 1:
                array = applyrandompolynomial(array, self.polyvars, self.polydegree, self.polyscale)
        if self.rotation == 0:
            rotationmatrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        else:
            rotationmatrix = ortho_group.rvs(dim=3)
        transformationmatrix = rotationmatrix + np.random.normal(0,self.transformation,(3,3))
        transformedpoints = np.dot(array*np.random.normal(self.scale,self.scale_randomness,1), transformationmatrix) + np.random.normal(0,self.translation,3) + np.random.normal(0,self.noise,array.shape)
        return transformedpoints


    def set_transforms(self, rotation=0, scale=4, scale_randomness=0, noise=0, translation = 0, transformation = 0, polyvars = 2, polydegree = 2, polyscale = 0, polycoef = [[0,0,1],[0,0,0],[1,0,0]], polyrandom = 0):
        self.noise = noise
        self.scale = scale
        self.scale_randomness = scale_randomness
        self.translation = translation
        self.transformation = transformation
        self.rotation = rotation
        self.polyvars = polyvars
        self.polydegree = polydegree
        self.polyscale = polyscale
        self.polycoef = polycoef
        self.polyrandom = polyrandom

    def as_grid(self, grid_space, enforce = True, sampling_floats=0):
        """
        grid is gridspace object
        """
        grid = grid_space.get_empty()
        for point in self.points:
            index = grid_space.get_index(point)
            if index != None:
                if sampling_floats:
                    grid[index] += 1
                else:
                    grid[index] = 1
            elif enforce != True:
                pass
            else:
                raise IOError("{0} not in bounds, exiting".format(index))
                print("{0} not in bounds, omitting".format(index))
        if sampling_floats:
            return grid/float(np.amax(grid))
        else:
            return grid
        # Iterate through the voxels of the grid and run self.is_inside() on
        # the point, turning the point to 1 if true

    def is_inside(self, point):
        return False


class Sphere(Shape):
    def __init__(self, classarray):
        Shape.__init__(self, classarray)
        self.radius = 1

    def sample(self, n):
        points = []
        for i in range(n):
            randpoint = np.array([random.gauss(0, 1) for k in range(3)])
            magnitude = float(
                math.sqrt(sum([math.pow(term, 2) for term in randpoint])))
            spherepoint = randpoint / magnitude * self.radius
            points.append(spherepoint)
        self.points = self.transform(np.array(points))
        return self.points


class Torus(Shape):
    def __init__(self, classarray):
        Shape.__init__(self, classarray)
        self.minor_radius = .3
        self.major_radius = 1

    def sample(self, n):
        points = []
        for i in range(n):
            u = np.random.uniform(0, 2 * math.pi)
            v = self.torusrejectionsampling()
            x = (self.major_radius + self.minor_radius * math.cos(v)) * math.cos(u)
            y = (self.major_radius + self.minor_radius * math.cos(v)) * math.sin(u)
            z = self.minor_radius * math.sin(v)
            toruspoint = [x, y, z]
            points.append(toruspoint)
        self.points = self.transform(np.array(points))
        return self.points

    #For uniform sampling from torus - from the Diaconis paper

    def torusrejectionsampling(self):
        x = np.random.uniform(0,2 * np.pi)
        y = np.random.uniform(0,1 / np.pi)
        fx = (1 + (self.minor_radius / self.major_radius)*np.cos(x))/(2* np.pi)
        while y > fx:
            x = np.random.uniform(0,2 * np.pi)
            y = np.random.uniform(0,1 / np.pi)
            fx = (1 + (self.minor_radius / self.major_radius)*np.cos(x))/(2* np.pi)
        return x

class Pyramid(Shape):
    def __init__(self, classarray):
        Shape.__init__(self, classarray)
        self.side_length = 1

    def sample(self, n):
        points = []

        e1 = np.array([self.side_length, 0, 0])
        e2 = np.array([0, self.side_length, 0])
        e3 = np.array([0, 0, self.side_length])

        i = 0
        while i < n:
            face = random.uniform(0, 4)
            if face < 1:
                #base (e1,e2)
                a = random.uniform(0, 1)
                b = random.uniform(0, 1)
                while (a + b) > 1:
                    a = random.uniform(0, 1)
                    b = random.uniform(0, 1)
                point = a * e1 + b * e2
            else:
                if face < 2:
                    # face 1(e1,e3)
                    a = random.uniform(0, 1)
                    b = random.uniform(0, 1)
                    while (a + b) > 1:
                        a = random.uniform(0, 1)
                        b = random.uniform(0, 1)
                    point = a * e1 + b * e3
                else:
                    if face < 3:
                        # face 2(e2,e3)
                        a = random.uniform(0, 1)
                        b = random.uniform(0, 1)
                        while (a + b) > 1:
                            a = random.uniform(0, 1)
                            b = random.uniform(0, 1)
                        point = a * e2 + b * e3
                    else:
                        # face 3
                        a = random.uniform(0, 1)
                        b = random.uniform(0, 1)
                        while (a + b) > 1:
                            a = random.uniform(0, 1)
                            b = random.uniform(0, 1)
                        c = 1 - (a + b)
                        point = a * e1 + b * e2 + c * e3
            points.append(point)
            i = i + 1
        self.points = self.transform(np.array(points))
        return self.points


class Cube(Shape):
    def __init__(self, classarray):
        Shape.__init__(self, classarray)
        self.side_length = 1

    def sample(self, n):
        points = []

        e1 = np.array([self.side_length, 0, 0])
        e2 = np.array([0, self.side_length, 0])
        e3 = np.array([0, 0, self.side_length])

        i = 0
        while i < n:
            face = random.choice([1, 2, 3, 4, 5, 6])
            if face == 1:
                #base (e1,e2)
                a = random.uniform(0, 1)
                b = random.uniform(0, 1)
                point = a * e1 + b * e2
            if face == 2:
                # face 1(e1,e3)
                a = random.uniform(0, 1)
                b = random.uniform(0, 1)
                point = a * e1 + b * e3
            if face == 3:
                # face 2(e2,e3)
                a = random.uniform(0, 1)
                b = random.uniform(0, 1)
                point = a * e2 + b * e3
            if face == 4:
                #base (e1,e2) + zhat
                a = random.uniform(0, 1)
                b = random.uniform(0, 1)
                point = a * e1 + b * e2 + np.array([0, 0, self.side_length])
            if face == 5:
                # face 1(e1,e3) + yhat
                a = random.uniform(0, 1)
                b = random.uniform(0, 1)
                point = a * e1 + b * e3 + np.array([0, self.side_length, 0])
            if face == 6:
                # face 2(e2,e3) + xhat
                a = random.uniform(0, 1)
                b = random.uniform(0, 1)
                point = a * e2 + b * e3 + np.array([self.side_length, 0, 0])

            points.append(point)
            i = i + 1
        self.points = self.transform(np.array(points))
        return self.points


class SquareTorus(Shape):
    def __init__(self, classarray):
        Shape.__init__(self, classarray)
        self.outside_side_length = 1
        self.inside_side_length = .3
        self.height = .8

    def sample(self, n):
        points = []

        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])

        i = 0
        while i < n:

            face = random.choice([1, 2, 3, 4, 5])
            a = random.uniform(0,1)*random.choice([-1,1])
            b = random.uniform(0,1)*random.choice([-1,1])
            # Outer Faces
            if face == 1:
                point = .5*e1*random.choice([-1,1])*self.outside_side_length + .5*self.height*e2*a + .5*self.outside_side_length*e3*b
            elif face == 2:
                point = .5*e3*random.choice([-1,1])*self.outside_side_length + .5*self.height*e2*a + .5*self.outside_side_length*e1*b
            elif face == 3:
                point = .5*e1*random.choice([-1,1])*self.inside_side_length + .5*self.height*e2*a + .5*self.inside_side_length*e3*b
            elif face == 4:
                point = .5*e3*random.choice([-1,1])*self.inside_side_length + .5*self.height*e2*a + .5*self.inside_side_length*e1*b
            elif face == 5:
                valid = False
                while valid == False:
                    a = random.uniform(0,1)*random.choice([-1,1])
                    b = random.uniform(0,1)*random.choice([-1,1])
                    term1 = .5*e2*random.choice([-1,1])*self.height
                    term2 = .5*self.outside_side_length*a #e1
                    term3 = .5*self.outside_side_length*b #e3
                    if (abs(term2) >= abs(.5*self.inside_side_length)) or (abs(term3) >= abs(.5*self.inside_side_length)):
                        point = term1+e1*term2+e3*term3
                        valid = True
            points.append(point)
            i = i + 1
        self.points = self.transform(np.array(points))
        return self.points
