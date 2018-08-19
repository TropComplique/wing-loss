import numpy as np


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    It aligns a shape defined by Y to a shape defined by X.
    # Y_transformed = b * YT + c

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

    Arguments:
        X, Y: float numpy arrays with shape [n, p].
        scaling: a boolean, if False, the scaling component of the transformation is forced
        to 1
        reflection: a string or boolean,
            possible values are 'best', False, True.
            if 'best' (default), the transformation solution may or may not
            include a reflection component, depending on which fits the data
            best. setting reflection to True or False forces a solution with
            reflection or no reflection respectively.
    Returns:
        Y_transformed: a float numpy array with shape [n, p].
        a dict specifying the rotation, translation and scaling that
        maps X --> Y
    """
    muX = X.mean(0)
    muY = Y.mean(0)

    # center shapes
    X0 = X - muX
    Y0 = Y - muY

    # compute centered frobenius norm
    normX = np.sqrt((X0**2).sum())
    normY = np.sqrt((Y0**2).sum())

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    # get an optimal rotation matrix of Y
    A = np.matmul(X0.T, Y0)  # shape [p, p]
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # they have shapes [p, k], [k], [k, p]
    V = Vt.T
    T = np.matmul(V, U.T)  # shape [p, p]
    # T is orthogonal

    if reflection is not 'best':

        # does the current solution use a reflection?
        has_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != has_reflection:
            V[:, -1] *= -1
            s[-1] *= -1  # the smallest singular value
            T = np.matmul(V, U.T)

    traceTA = s.sum()  # trace of TA

    if scaling:

        # optimal scaling of Y
        b = traceTA * normX / normY

        # transformed coords
        Z = normX * traceTA * np.matmul(Y0, T) + muX

        # frobenius_norm(Y0 * T) = frobenius_norm(Y0) = 1

    else:
        b = 1
        Z = normY * np.matmul(Y0, T) + muX

    c = muX - b * np.matmul(muY, T)

    transform = {'rotation': T, 'scale': b, 'translation': c}
    """
    b * Y * T + c =
    = b * (Y0 * normY + muY) * T + muX - b * muY * T =
    = b * normY * Y0 * T + muX =
    = Z
    """
    return Z, transform
