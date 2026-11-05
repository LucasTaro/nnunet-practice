# from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
import numpy as np

def compute_new_shape(
    old_shape,
    old_spacing,
    new_spacing
):
    new_shape = np.array([
        int(round(i / j * k))
        for i, j, k in zip(
            old_spacing,
            new_spacing,
            old_shape
        )
    ])
    return new_shape

if __name__ == "__main__":
    print(
        compute_new_shape(
            [512, 512, 54],
            [0.4688, 0.4688, 7.0],
            [0.4688, 0.4688, 2.0]
        )
    )