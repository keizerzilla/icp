from registration import *

def load_xyz(file_path):
    return np.loadtxt(file_path, delimiter=" ")


def save_xyz(file_path, C):
    return np.savetxt(file_path, C)


if __name__ == "__main__":
    A = load_xyz("clouds/bunny_head.xyz")
    B = load_xyz("clouds/bunny_transform.xyz")
    
    R, t, i = icp(A, B)
    
    print(R)
    print(t)
    print(i)
    
    new_cloud = np.matmul(A, R.T) + t
    save_xyz("teste.xyz", new_cloud)
    