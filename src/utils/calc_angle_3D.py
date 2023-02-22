import numpy as np

import math


def calculate_angle_3D(a:np.ndarray,b:np.ndarray,c:np.ndarray):
    assert len(a) == len(b) == 3
    
    ba = a-b
    bc = c-b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)



def calculate_foot_direction(ancle:np.ndarray, knee:np.ndarray, hip:np.ndarray)->np.ndarray:
    """calculate direction of foot with the direction of the area between hip-anke-knee"""
    
    vector_ah = vector(ancle,hip)
    length_hip_knee = np.linalg.norm(vector(hip,knee))
    length_of_dir = pnt2line(knee,ancle,hip)
    length_midpoint = math.sqrt(length_hip_knee**2 - length_of_dir[0]**2)
    length_ah = np.linalg.norm(vector_ah)
    midpoint = ancle + np.true_divide(np.array(vector_ah),length_ah) * length_midpoint
    pointer = knee-midpoint
    return pointer
    

#neares distance between point and line 
def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)  
    
#helpers


def dot(v,w):
    x,y,z = v
    X,Y,Z = w
    return x*X + y*Y + z*Z

def length(v):
    x,y,z = v
    return math.sqrt(x*x + y*y + z*z)

def vector(b,e):
    x,y,z = b
    X,Y,Z = e
    return (X-x, Y-y, Z-z)

def unit(v):
    x,y,z = v
    mag = length(v)
    return (x/mag, y/mag, z/mag)

def distance(p0,p1):
    return length(vector(p0,p1))

def scale(v,sc):
    x,y,z = v
    return (x * sc, y * sc, z * sc)

def add(v,w):
    x,y,z = v
    X,Y,Z = w
    return (x+X, y+Y, z+Z)


if __name__ == "__main__":
    ancle = np.array([1,0,0])
    knee = np.array([2,1,0])
    hip = np.array([1,2,0])
    calculate_foot_direction(ancle,knee,hip)