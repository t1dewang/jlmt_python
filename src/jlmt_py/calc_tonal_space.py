import numpy as np
from scipy.io import loadmat

def load_som(som_path: str):
    """Read SOM.mat file containing sM and return (weights, msize)."""
    mat = loadmat(som_path, squeeze_me=True, struct_as_record=False)
    if "sM" not in mat:
        raise ValueError(f"There is no 'sM' in {som_path}.keys={list(mat.keys())}")
    sM = mat["sM"]

    # take codebook（n_units, n_features）
    weights = getattr(sM, "codebook", None)
    if weights is None:
        raise ValueError("SOM file is missing 'sM.codebook'.")
    weights = np.asarray(weights)

    # take topological size msize (rows, cols)
    topol = getattr(sM, "topol", None)
    if topol is None:
        raise ValueError("SOM file is missing 'sM.topol'.")
    msize = getattr(topol, "msize", None)
    if msize is None:
        try:
            msize = topol["msize"]
        except Exception as e:
            raise ValueError("SOM file is missing 'sM.topol.msize'.") from e
    msize = tuple(np.array(msize, dtype=int).ravel())

    return weights, msize


def calc_tonal_space(li, weights, msize):
    """
    Projects li images into Tonal Space using the SOM weight matrix.
   
    Parameters:
        li (np.ndarray): Leaky-integrated images of shape (n_features, n_frames)
        weights (np.ndarray): SOM weight matrix of shape (n_units, n_features).
        msize (tuple): Topology of the SOM grid given as (rows, cols).

    Returns:
        ts_activation_map_3d (np.ndarray): 3D Tonal Space activation map of shape (rows, cols, n_frames)
        ts_activation_map_2d (np.ndarray): 2D Tonal Space activation map of shape (n_units, n_frames)
    """
    print(f"Computing Tonal Space projection...")
    
    n_features_li, n_frames = li.shape
    n_units, n_features_weights = weights.shape

    if n_features_li != n_features_weights:
        raise ValueError(
            f"Feature dimension mismatch: LI has {n_features_li} features, "
            f"SOM weights have {n_features_weights}."
        )

    # 1. Normalization
    
    # Sum over each time frame (column)
    column_sums = li.sum(axis=0, keepdims=True)
    
    # Normalize each column; avoid division by zero using np.divide 'where'
    limg_norm = np.divide(li, column_sums, out=np.zeros_like(li), where=column_sums!=0)

    # 2. Projection onto SOM
    ts_activation_map_2d = weights @ limg_norm

    # 3. Reshape into 3D map (
    rows, cols = msize
    n_frames_act = ts_activation_map_2d.shape[1] 
    
    ts_activation_map_3d = ts_activation_map_2d.reshape((rows, cols, n_frames_act), order='F')
    
    print(f"...Done. TS activation map shape: {ts_activation_map_3d.shape}")
    
    return ts_activation_map_3d, ts_activation_map_2d