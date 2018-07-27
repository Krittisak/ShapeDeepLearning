##This modifies the pointlist by sampling uniformly at random from a group conjugate to SO(3) by a polynomial
##The polynomial can be random :applyrandompolynomial(pointlist,vars,degree,scalefactor)
##Or not: def applyfixedpolynomial(pointlist,vars,coef,scalefactor):

#The program produces a polynomial g(x,y) and picks a random rotation A, then defines f: (x,y,z) -> (x,y,z+ g(x*scalefactor,y*scalefactor)), and applies f A f^{-1} to the pointcloud. This will follow the random linear transformation in the shape class, so that the apparent dependence on the x,y,z coordinates disappears. Other

##The parameters here are:

##vars == 1 or vars == 2 ... how many variables the polynomial depends on
##degree -- the degree of thepolynomial
##coef = an array of coefficients : for vars == 1, is of form [1,4,2] for 1 + 4x^2 + 2x^3
##for vars == 2 ... coefficient of x^i y^j should be coef[i,j]... but this seems weirdly finicky, i think you need to put in a matrix of coefficients ... so ar = [[1,0,0],[0,0,1]] would correspond to 1 + xy^2 and p2d(x0,y0, ar) will output 1 + x0 *(y0)^2

##scale factor -- the conjugates the transformation by scaling by scalefactor. When the scalefactor is 0, will throw a divide by zero. not that random polynomials have a constant term currently, which acts like a translation.
import numpy as np
from numpy.polynomial import Polynomial as P
from numpy.polynomial.polynomial import polyval2d as p2d
from scipy.linalg import qr

def resize(pointlist):
	#nn takes in points from cube of dimensions... in config file
	##just find max and min point values and rescale window accordingly.
	return pointlist

def scale(pointlist, scalefactor):
	for term in pointlist:
		term[0] = term[0]*scalefactor
		term[1] = term[1]*scalefactor
		term[2] = term[2]*scalefactor
	return pointlist


def applyrandompolynomial(pointlist,vars,degree,scalefactor):
	pointlist = scale(pointlist, scalefactor)
	if vars == 1:
		randomcoef = np.random.randn(degree + 1)
		p = P(randomcoef)
		for term in pointlist:
			term[2] = term[2] + p(term[0])
		pointlist = randomrot(pointlist)
		for term in pointlist:
			term[2] = term[2] - p(term[0])
	if vars == 2:
		randomcoef = np.random.randn(degree,degree)
		for term in pointlist:
			term[2] = term[2] + p2d(term[0],term[1],randomcoef)
		pointlist = randomrot(pointlist)
		for term in pointlist:
			term[2] = term[2] - p2d(term[0], term[1], randomcoef)
	pointlist = scale(pointlist, 1 / float(scalefactor))
	return pointlist

def applyquadratic(pointlist,scalefactor):
	#for debugging purposes, should be subsumed in applyfixedpolynomial
	pointlist = scale(pointlist, scalefactor)
	for term in pointlist:
		term[2] = term[2] + np.power(term[0],2) + np.power(term[1],2)
	pointlist = randomrot(pointlist)
	for term in pointlist:
		term[2] = term[2] - np.power(term[0],2) - np.power(term[1],2)
	pointlist = scale(pointlist, 1/ float(scalefactor))
	return pointlist

def randomrot(pointlist):
	H = np.random.randn(3,3)
	Q,R = qr(H)
	pointlist = np.dot(pointlist, Q)
	return pointlist

def applyfixedpolynomial(pointlist,vars,coef,scalefactor):
	pointlist = scale(pointlist, float(scalefactor))
	pointlist = applyg(pointlist,vars,coef)
	pointlist = randomrot(pointlist)
	pointlist = applyginv(pointlist, vars, coef)
	pointlist = scale(pointlist, 1/ float(scalefactor))
	return pointlist

def applyg(pointlist,vars,coef):
	if vars == 1:
		p = P(coef)
		for term in pointlist:
			term[2] = term[2] + p(term[0])
	if vars == 2:
		for term in pointlist:
			term[2] = term[2] + p2d(term[0],term[1],coef)

	return pointlist

def applyginv(pointlist,vars,coef):
	if vars == 1:
		p = P(coef)
		for term in pointlist:
			term[2] = term[2] - p(term[0])
	if vars == 2:
		for term in pointlist:
			term[2] = term[2] - p2d(term[0], term[1], coef)

	return pointlist
