import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from math import factorial

random.seed(0)


#Produces a line that traverses the given points -in order-
def interpolate(pts):
    plt.figure()
    plt.title("Interpolation")
    numPts = pts.shape[1]
    plt.scatter(pts[0], pts[1])
    
    freq = 1.0 / (numPts - 1)
    
    mat = np.array(
        [[(j * freq) ** i 
          for i in range(numPts)]
         for j in range(numPts)])
    
    coeff = np.matmul(np.linalg.inv(mat), pts.T)
     
    delta = 0.001
    t_mat = np.array([[(delta * j) ** i for i in range(numPts)]
                      for j in range(int(1/delta) + 1)])
    x = np.matmul(t_mat, coeff)
    plt.plot(x[:,0], x[:,1], 'r-')
    
    plt.savefig("./Plots/Interpolation.png", dpi=300)
    
     
def hermiteSplines(pts):
    plt.figure()
    plt.title("Hermite Splines")
    numPts = pts.shape[1]
    colours = ["r-", "g-", "b-"]
    
    delta = 0.1
    
    for i in range(0, pts.shape[1], 4):
        plt.scatter(pts[0,i:i + 2], pts[1,i:i + 2])
        mat = np.array([[0, 0, 0, 1],
                        [1, 1, 1, 1],
                        [0, 0, 1, 0],
                        [3, 2, 1, 0]])
    
        coeff = np.matmul(np.linalg.inv(mat), pts[:,i:i + 4].T)
        print coeff
        t_mat = np.array([[(delta * k) ** j for j in range(numPts)]
                          for k in range(int(1/delta) + 1)])
        
        print t_mat
        x = np.matmul(t_mat, coeff)
        print x
        plt.plot(x[:,0], x[:,1], colours[i % 3])
    
    plt.savefig("./Plots/HermiteSplines.png", dpi=300) 
    
     
def bezierCurve(pts):
    plt.figure()
    plt.title("Bezier Curve")
    plt.scatter(pts[0], pts[1])
    numPts = pts.shape[1]
    
    delta = 0.1
    
    t = [delta * i for i in range(int(1/delta) + 1)]
    c_t = []
    for i in t:
        c_t.append(sum([pts[:,j] * bernsteinTerm(numPts - 1, j, i) for j in range(numPts)]))
    
    c_t = np.array(c_t)
    print c_t
    
    plt.plot(c_t[:,0], c_t[:,1], 'r-')
        
    plt.savefig("./Plots/BezierCurve.png", dpi=300) 
        

def bernsteinTerm(N, i, t):
    return ((factorial(N) / (factorial(N - i) * factorial(i))) *
            ((1 - t) ** (N - i)) *
            (t ** i))
        
pts = np.array(
    [[1, 2, 3, 4, 6, 8, 10, 12],
     [1, -2, 3, -4, 6, 7, 3, 1]])    

#interpolate(pts)

pts = np.array(
    [[1, 2, 3],
     [1, -2, 3]])  

bezierCurve(pts)

#pts = np.array(
    #[[0,  1, 0, 1],
     #[0,  1, -1, 0]]) 

#hermiteSplines(pts)


    

    