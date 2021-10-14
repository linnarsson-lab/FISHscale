import numpy as np
from numba import jit, njit
import numba
from typing import Any, Generator, Tuple


def close_polygon(self, polygon: np.ndarray):
    """Check if polygon is closed. If not returns closed polygon.
    
    For a closed polygon the first and last point are identical.

    Args:
        polygon (np.ndarray): Array with shape (X,2).

    Returns:
        [np.ndarray]: [description]
    """
    if np.all(polygon[0] != polygon[-1]):
        polygon = np.vstack((polygon, polygon[0]))
        
    return polygon


@jit(nopython=True)
def is_inside_sm(polygon: np.ndarray, point: np.ndarray) -> Any:
    """Test if point is inside a polygon.

    From: https://github.com/sasamil/PointInPolygon_Py/blob/master/pointInside.py
    From: https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python

    Args:
        polygon (np.ndarray: Array of polygon coordinates. Polygons should 
            be closed, meaning that the first and last point are identical.
        point (np.ndarray): Array with X and Y coordinate of point 
            to query.

    Returns:
        [bool]: True if point is inside polygon.

    """
    length = len(polygon)-1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii<length:
        dy  = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy<0 or dy2<0:
                F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

                if point[0] > F: # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F: # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2==0 and (point[0]==polygon[jj][0] or (dy==0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0])<=0)):
                return 2

        ii = jj
        jj += 1

    return intersections & 1  

@njit(parallel=True)
def is_inside_sm_parallel(polygon: np.ndarray, points: np.ndarray, n_jobs=-1) -> np.ndarray:
    """Test if array of point is inside a polygon.

    Paralelized the points using numba. 

    From: https://github.com/sasamil/PointInPolygon_Py/blob/master/pointInside.py
    From: https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python

    Args:
        polygon (np.ndarray: Array of polygon coordinates. Polygons should 
            be closed, meaning that the first and last point are identical.
        point (np.ndarray): Array with X and Y coordinates of points 
            to query.
        n_jobs (int, optional): Number of threads Numba can use. If -1 it uses
            all threads. After execution of the function it returns to the
            starting value. Defaults to -1.

    Returns:
        [np.ndarray]: Boolean array, with True if point falls inside
            polygon.

    """
    #Set number of threads
    if n_jobs != -1:
        default_n_threads = numba.get_num_threads()
        numba.set_num_threads(n_jobs)
    
    #Perform function    
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean) 
    for i in numba.prange(ln):
        D[i] = is_inside_sm(polygon, points[i])
    
    #Reset number of threads
    if n_jobs != -1:
        numba.set_num_threads(default_n_threads)    
    
    return D  

def get_bounding_box(polygon: np.ndarray) -> np.ndarray:
    """Get the bounding box of a single polygon.

    Args:
        polygon (np.ndarray): Array of X, Y coordinates of the polython as
            columns.

    Returns:
        np.ndarray: Array with the Left Bottom and Top Right corner 
            coordinates: np.array([[X_BL, Y_BL], [X_TR, Y_TR]])

    """
    xmin, ymin = polygon.min(axis=0)
    xmax, ymax = polygon.max(axis=0)
    return np.array([[xmin, ymin], [xmax, ymax]])

def make_bounding_box(polygon_points: dict, simple: bool = True) -> dict:
    """Find bounding box coordinates of dictionary with polygon points.

    Args:
        polygon_points (dict): Dictionary with labels of each (multi-) 
            polygon and a list of (sub-)polygon(s) as numpy arrays.
        simple (bool, optional): True for a single polygon, False for a list
            of polygons, representing a multi-polygon. Defaults to True.

    Returns:
        dict: Dictionary in the same shape as input, but for every (sub-)
            polygon the bounding box.
    
    """
    results = {}
    labels = list(polygon_points.keys())
    #Loop over labels
    for l in labels:
        if simple:
            results[l] = get_bounding_box(polygon_points[l])
        else:
            results[l] = []
            #Loop over polygon list
            for poly in polygon_points[l]:
                results[l].append(get_bounding_box(poly))
    return results

def bbox_filter_points(bbox: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Filter point that fall within bounding box.

    Args:
        bbox (np.ndarray): Array with the Left Bottom and Top Right corner 
            coordinates: np.array([[X_BL, Y_BL], [X_TR, Y_TR]])
        points (np.ndarray): Array with X and Y coordinates of the points
            as columns

    Returns:
        np.ndarray: Boolean array with True for points that are in the 
            bounding box.

    """
    filt_x = np.logical_and(points[:,0]>=bbox[0][0], points[:,0]<=bbox[1][0])
    filt_y = np.logical_and(points[:,1]>=bbox[0][1], points[:,1]<=bbox[1][1])
    return np.logical_and(filt_x, filt_y)

def inside_simple_polygons(polygon_points: dict, points: np.ndarray) -> Generator[Tuple[Any, np.ndarray], None, None]:
    """Generator function that yield which points are in a polygon.

    Generator that yields a boolean array for the points that are inside a
    simple polygon. 
    Use the "inside_multi_polygons()" fucntion if you have polygons that have 
    more than one polygon (holes or multiple seperate polygons.)
    Function is paralelized.

    Args:
        polygon_points (dict): Dictionary with labels as keyes and a numpy 
            array with polygon points. Polygon needs to be closed, meaning 
            that the first and last points are identical.
        points (np.ndarray): Array with X and Y coordinates of the points
                as columns

    Yields:
        Generator[Tuple[Any, np.ndarray], None, None]: The label and a boolean
            filter for which points are in the polygon with that label.

    """
    bbox = make_bounding_box(polygon_points, simple = True)    
    labels = list(polygon_points.keys())

    #Iterate over all polygons
    for l in labels:            
        point_inside = np.zeros(points.shape[0]).astype('bool')
        #Filter points with the bounding box of the polygon
        filt = bbox_filter_points(bbox[l], points)
        #Check which points are inside the polygon
        is_inside = is_inside_sm_parallel(polygon_points[l], points[filt])
        point_inside[filt] = is_inside
        
        yield l, point_inside

def inside_multi_polygons(polygon_points: dict, points: np.ndarray) -> Generator[Tuple[Any, np.ndarray], None, None]:
    """Generator function that yield which points are in a multi-polygon.

    Generator that yields a boolean array for the points that are inside a
    complex polygon. A complex polygon is given as a list of simple polygons,
    for all seperat areas and holes in the polygons. 
    Function is paralelized.

    Args:
        polygon_points (dict): Dictionary with labels of each (multi-) 
            polygon and a list of (sub-)polygon(s) as numpy arrays. Polygon 
            needs to be closed, meaning that the first and last points are 
            identical.
        points (np.ndarray): Array with X and Y coordinates of the points
                as columns

    Yields:
        Generator[Tuple[Any, np.ndarray], None, None]: The label and a boolean
            filter for which points are in the polygon with that label.

    """
    bbox = make_bounding_box(polygon_points, simple=False)    
    labels = list(polygon_points.keys())

    #Iterate over all (multi) polygons
    for l in labels:    
        point_inside = np.zeros(points.shape[0]).astype('bool')
        
        #Iterate over every (sub) polygon of the (multi) polygon
        for p, bb in zip(polygon_points[l], bbox[l]):
            #Filter points with the bounding box of the polygon
            filt = bbox_filter_points(bb, points)
            #Check which points are inside the polygon
            is_inside = is_inside_sm_parallel(p, points[filt])
            #For a point to be inside a multi polygon, it needs to be found inside the sub-polygons an uneven number 
            #of times. If a point is inside one sub-polygon, it is inside. If it is inside two sub-polygons it 
            #means that it is outside the multi-poligon because the second polygon has to be inside the first. If it
            #is in 3 sub-polygons it means there is a large polygon with a hole in it, made by the second polygon,
            #which has a 3rd polygon inside, that contains the point.
            point_inside[filt] = np.logical_xor(point_inside[filt], is_inside)

        yield l, point_inside