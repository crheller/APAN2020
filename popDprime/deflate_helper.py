

def deflate_noise(x, axis):
    """
    Deflate out axis just for the noise. i.e., don't operate on 
    the mean of the stimulus.
    x is reps X dim
    """
    _x = x - x.mean(axis=0)
    proj = (_x.dot(axis) @ axis.T)
    _x = _x - proj
    x = x.mean(axis=0) + _x
    return x