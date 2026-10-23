# Write Function to evaluate 1D basis functions and their derivatives over a single line segment

# Triangular B-Spline function over a line segment 
import numpy as np
def basis1d(nodes, pts, direction):
    # nodes is a list with node1 and node2 as elements
    # pts is a list of points to evaluate
    # direction is and int 1 or 2, 1 is increasing function, 2 is decreasing
    
    n1 = nodes[0] # node 1
    n2 = nodes[1] # node 2
    # func is function value at point
    # deriv is derivative function value at point

    fvals = [] # fvals is a list of the function values for the given points
    derivvals = [] # derivvals is a list of the derivative function values 

    for i in range(len(pts)): 
        if not direction: # increasing 
            ans = (n2 - pts[i]) / (n2 - n1)
            deriv = -1 / (n2 - n1) 
            
        else: # decreasing
            ans = (pts[i] - n1) / (n2 - n1)
            deriv = 1 / (n2 - n1)
        fvals.append(ans)
        derivvals.append(deriv)
    
    return np.array(fvals), np.array(derivvals)
