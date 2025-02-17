import itertools
import numpy as np

def get_inverse_permutation(permutation):
    return np.argsort(permutation)

random_shape = np.random.randint(1, 10, (5,))
random_array = np.random.rand(*random_shape)

for permutation in itertools.permutations([0, 1, 2, 3, 4], 5):
    permuted_array = np.transpose(random_array, permutation)
    inverse_permutation = get_inverse_permutation(permutation)
    original_array = np.transpose(permuted_array, inverse_permutation)
    assert np.array_equal(random_array, original_array), f"{random_array.shape} != {original_array.shape}"
