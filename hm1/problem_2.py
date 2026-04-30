import numpy as np
from kin_func_skeleton import prod_exp

#########################################
#               EXAMPLE                 #
#########################################


def scara_fk(theta):
    """
    An example implementation of a forward kinematics map.
    Feel free to use this as a template for your own implementations
    in this file.

    This function implements the forward kinematics map of the
    SCARA manipulator, following Example 3.1 from MLS (page 87).

    We take L0 = L1 = L2 = 1

    Arguments:
        theta: numpy.ndarray of size (4,), the values of the joint angles.
               theta[i] is the value of the ith joint, at which the
               FK map should be computed.
    Returns:
        - g (numpy.ndarray of shape (4,4)): the 4x4 configuration of the
          end effector when the joints have been placed at the angles
          specified in theta.
        - xi_array (numpy.ndarray of shape (6, N)): an array with the twists
          stacked in its columns.
    """

    # Specify all twists.
    xi_1 = [0, 0, 0, 0, 0, 1]
    xi_2 = [1, 0, 0, 0, 0, 1]
    xi_3 = [2, 0, 0, 0, 0, 1]
    xi_4 = [0, 0, 1, 0, 0, 0]

    # Specify end effector configuration at theta = 0.
    gst0 = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 2], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.float64
    )

    # Stack twists into an array that prod_exp can accept.
    xi_array = np.array([xi_1, xi_2, xi_3, xi_4], dtype=np.float64).T

    # Use product of exponentials formula to compute forward kinematics.
    g = np.matmul(prod_exp(xi_array, theta), gst0)

    # Return the required quantities.
    return g, xi_array


#########################################
#              HW PROBLEMS              #
#########################################


def fk(theta):
    """
    Stanford arm forward kinematics (5R + 1P).

    Arguments:
        theta: (6,) array of joint values.

    Returns:
        - g: (4,4) end-effector configuration matrix.
        - xi_array: (6, 6) array of twists stacked as columns.
    """

    l0, l1 = 1.0, 1.0
    xi_1 = np.array([0, 0, 0, 0, 0, 1])
    xi_2 = np.array([0, -l0, 0, -1, 0, 0])
    xi_3 = np.array([0, 1, 0, 0, 0, 0])
    qw = np.array([0, l1, l0])
    xi_4 = np.hstack([-np.cross([0, 0, 1], qw), [0, 0, 1]])
    xi_5 = np.hstack([-np.cross([-1, 0, 0], qw), [-1, 0, 0]])
    xi_6 = np.hstack([-np.cross([0, 1, 0], qw), [0, 1, 0]])

    xi_array = np.array([xi_1, xi_2, xi_3, xi_4, xi_5, xi_6]).T

    # Initial config: end-effector at wrist center
    gst0 = np.eye(4)
    gst0[:3, 3] = qw

    print("gst0 =")
    print(gst0)

    for i in range(6):
        print(f"xi_{i+1} = {xi_array[:, i]}")

    print(f"Current theta = {theta}")
    g = prod_exp(xi_array, theta) @ gst0

    print("\nResult T_ST(theta) =")
    print(g)

    return g, xi_array


if __name__ == "__main__":
    theta_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    g, xi_array = fk(theta_test)
