import cv2
import numpy as np

##########################################################################

def array_norm(A):
    mn = A.min()
    mx = A.max()
    B = (A-mn)/(mx-mn)
    return B

        
#################################################################

def draw3D (r):
    x = range(r.shape[0])
    y = range(r.shape[1])

    data = np.array(r)

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, data)

    plt.show()
    
###########################################################
def save_array_csv(A, file_name):
    with open(file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerows(A)
        