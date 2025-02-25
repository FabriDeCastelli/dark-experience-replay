import numpy as np
    
def random_permutation(sample: np.ndarray) -> np.ndarray:
    """
    Randomly permutes the image.

    Args:
        sample: image to be permuted

    Returns:
        permuted image
    """
    perm = np.random.permutation(np.prod(sample.shape))
    old_shape = sample.shape

    return sample.flatten()[perm].reshape(old_shape)