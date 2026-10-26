"""
   Copyright 2026 Samuel Elkin, Ghazi Khan, Thomas Roth

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

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
