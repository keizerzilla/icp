import icp
import time


# number of random points in the dataset
N = 1000

# number of test iterations
num_tests = 100

# number of dimensions of the points
dim = 3

# standard deviation error to be added
noise_sigma = .01

# max translation of the test set
translation = .1

# max rotation (radians) of the test set
rotation = .1


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d),     2*(b*d+a*c)],
                     [2*(b*c+a*d),     a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c),     2*(c*d+a*b),     a*a+d*d-b*b-c*c]])


def test_best_fit():
    # Generate a random dataset
    A = np.random.rand(N, dim)
    
    total_time = 0
    for i in range(num_tests):
        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand()*rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Find best fit transform
        start = time.time()
        T, R1, t1 = icp.best_fit_transform(B, A)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = B

        # Transform C
        C = np.dot(T, C.T).T
        
        # T should transform B (or C) to A
        assert np.allclose(C[:,0:3], A, atol=6*noise_sigma)
        
        # t and t1 should be inverses
        assert np.allclose(-t1, t, atol=6*noise_sigma)
        
        # R and R1 should be inverses
        assert np.allclose(R1.T, R, atol=6*noise_sigma)

    print("best fit time: {:.3}".format(total_time/num_tests))


def test_icp():
    # Generate a random dataset
    A = np.random.rand(N, dim)

    total_time = 0
    for i in range(num_tests):
        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Shuffle to disrupt correspondence
        np.random.shuffle(B)

        # Run ICP
        start = time.time()
        T, distances, iterations = icp.icp(B, A, tolerance=0.000001)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = np.copy(B)

        # Transform C
        C = np.dot(T, C.T).T

        # mean error should be small
        assert np.mean(distances) < 6*noise_sigma
        
        # T and R should be inverses
        assert np.allclose(T[0:3,0:3].T, R, atol=6*noise_sigma)
        
        # T and t should be inverses
        assert np.allclose(-T[0:3,3], t, atol=6*noise_sigma)

    print("icp time: {:.3}".format(total_time/num_tests))


if __name__ == "__main__":
    test_best_fit()
    test_icp()

