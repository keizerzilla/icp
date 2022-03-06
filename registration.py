import numpy as np
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding
    points A to B in m spatial dimensions.
    
    Parameters
    ----------
    A: Nxm numpy array of corresponding points.
    B: Nxm numpy array of corresponding points.
    
    Returns
    -------
    R: mxm rotation matrix.
    t: mx1 translation vector.
    """
    
    # get number of dimensions
    m = A.shape[1]
    
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    
    # rotation matrix
    H = np.matmul(AA.T, B)
    U, S, V = np.linalg.svd(H)
    R = np.matmul(V.T, U.T)
    
    if np.linalg.det(R) < 0:
       V[m-1,:] *= -1
       R = np.matmul(V.T, U.T)
    
    # translation
    t = centroid_B - np.matmul(R, centroid_A)
    
    return R, t


def angle_norm(n, x):
    # Ângulos que os pontos de uma nuvem 'x' fazem com um plano de normal 'n'
    
    prod = n * x
    num = np.abs(prod.sum(axis=1))
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(n)
    
    return np.arcsin(num / den)


def cloud_preproc(cloud):
    # Converte nuvem cartesiana 'cloud' em representação RABG
    
    cloud = cloud - np.mean(cloud, axis=0)
    _, _, v = np.linalg.svd(cloud, full_matrices=False)
    angles = np.apply_along_axis(angle_norm, 1, v, cloud).T
    distances = np.linalg.norm(cloud, axis=1)
    rhos = distances / np.amax(distances)
    rhos = rhos.reshape((-1, 1)) # <- talvez nem precise, heim
    
    #return np.concatenate((rhos, angles), axis=1)
    return angles


def singular_dir(cloud):
    # @TODO
    
    cloud = cloud - np.mean(cloud, axis=0)
    _, _, v = np.linalg.svd(cloud, full_matrices=False)
    
    return v


def preproc_feat(cloud, v):
    # @TODO
    
    cloud = cloud - np.mean(cloud, axis=0)
    angles = np.apply_along_axis(angle_norm, 1, v, cloud).T
    #distances = np.linalg.norm(cloud, axis=1)
    #rhos = distances / np.amax(distances)
    #rhos = rhos.reshape((-1, 1)) # <- talvez nem precise, heim
    
    #return np.concatenate((rhos, angles), axis=1)
    return angles

def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src.
    
    Parameters
    ----------
    src: Nxm array of points.
    dst: Nxm array of points.
    
    Returns
    -------
    distances: Euclidean distances of the nearest neighbor.
    indices: dst indices of the nearest neighbor.
    """
    
    #a = cloud_preproc(src)
    #b = cloud_preproc(dst)
    
    v_d = singular_dir(dst)
    
    a = preproc_feat(src, v_d)
    b = preproc_feat(dst, v_d)
    
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(b)
    distances, indices = neigh.kneighbors(a, return_distance=True)
    
    return distances.ravel(), indices.ravel()


def icp(A, B, max_iterations=20, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps
    points A on to points B.
    
    Parameters
    ----------
    A: Nxm numpy array of source mD points.
    B: Nxm numpy array of destination mD point.
    max_iterations: exit algorithm after max_iterations.
    tolerance: convergence criteria.
    
    Returns
    -------
    R: mxm rotation matrix.
    t: mx1 translation vector.
    i: number of iterations to converge.
    """
    
    # get number of dimensions
    m = A.shape[1]
    
    # make points homogeneous, copy them to maintain the originals
    src = np.copy(A)
    dst = np.copy(B)
    
    prev_error = 0
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination
        # points
        distances, indices = nearest_neighbor(src, dst)
        
        # compute the transformation between the current source and nearest
        # destination points
        R, t = best_fit_transform(src, dst[indices])
        
        # update the current source
        src = np.matmul(src, R.T) + t
        
        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        
        prev_error = mean_error
    
    # calculate final transformation
    R, t = best_fit_transform(A, src)
    
    return R, t, i
