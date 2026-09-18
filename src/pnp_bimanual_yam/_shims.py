"""Import-time shims so mimicgen.utils.pose_utils can load without robosuite/gdown.

MimicGen normally depends on robosuite for pose helpers and gdown for optional
dataset downloads. In this bridge we never touch robosuite envs; MimicGen only
needs mat2quat/quat2mat scalar helpers and no network downloads.

Import this module BEFORE any mimicgen submodule.
"""

from __future__ import annotations

import sys
import types

import numpy as np


def _mat2quat(M: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> xyzw quaternion. Matches robosuite semantics."""
    M = np.asarray(M, dtype=float).reshape(3, 3)
    tr = M[0, 0] + M[1, 1] + M[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (M[2, 1] - M[1, 2]) / s
        qy = (M[0, 2] - M[2, 0]) / s
        qz = (M[1, 0] - M[0, 1]) / s
    else:
        i = int(np.argmax([M[0, 0], M[1, 1], M[2, 2]]))
        if i == 0:
            s = np.sqrt(max(0.0, 1.0 + M[0, 0] - M[1, 1] - M[2, 2])) * 2.0
            qw = (M[2, 1] - M[1, 2]) / s if s > 1e-12 else 1.0
            qx = 0.25 * s
            qy = (M[0, 1] + M[1, 0]) / s if s > 1e-12 else 0.0
            qz = (M[0, 2] + M[2, 0]) / s if s > 1e-12 else 0.0
        elif i == 1:
            s = np.sqrt(max(0.0, 1.0 + M[1, 1] - M[0, 0] - M[2, 2])) * 2.0
            qw = (M[0, 2] - M[2, 0]) / s if s > 1e-12 else 1.0
            qx = (M[0, 1] + M[1, 0]) / s if s > 1e-12 else 0.0
            qy = 0.25 * s
            qz = (M[1, 2] + M[2, 1]) / s if s > 1e-12 else 0.0
        else:
            s = np.sqrt(max(0.0, 1.0 + M[2, 2] - M[0, 0] - M[1, 1])) * 2.0
            qw = (M[1, 0] - M[0, 1]) / s if s > 1e-12 else 1.0
            qx = (M[0, 2] + M[2, 0]) / s if s > 1e-12 else 0.0
            qy = (M[1, 2] + M[2, 1]) / s if s > 1e-12 else 0.0
            qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def _quat2mat(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion -> 3x3 rotation matrix. Matches robosuite semantics."""
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def install() -> None:
    """Register fake robosuite / gdown / robomimic optional modules if absent."""
    if "robosuite" not in sys.modules:
        rs = types.ModuleType("robosuite")
        rs_utils = types.ModuleType("robosuite.utils")
        rs_T = types.ModuleType("robosuite.utils.transform_utils")
        rs_T.mat2quat = _mat2quat
        rs_T.quat2mat = _quat2mat
        rs_utils.transform_utils = rs_T
        rs.utils = rs_utils
        sys.modules["robosuite"] = rs
        sys.modules["robosuite.utils"] = rs_utils
        sys.modules["robosuite.utils.transform_utils"] = rs_T

    if "gdown" not in sys.modules:
        gd = types.ModuleType("gdown")

        def _download_unavailable(*_a, **_k):
            raise RuntimeError("gdown.download unavailable in bimanual-yam bridge")

        gd.download = _download_unavailable
        sys.modules["gdown"] = gd

    if "robomimic.utils.lang_utils" not in sys.modules:
        lu = types.ModuleType("robomimic.utils.lang_utils")
        lu.LANG_EMB_OBS_KEY = "*"

        def _unavailable(*_a, **_k):
            raise RuntimeError("language embeddings unavailable in bimanual-yam bridge")

        lu.get_lang_emb = _unavailable
        lu.get_lang_emb_shape = _unavailable
        sys.modules["robomimic.utils.lang_utils"] = lu

    if "robomimic.algo" not in sys.modules:
        algo = types.ModuleType("robomimic.algo")

        def _unavailable(*_a, **_k):
            raise RuntimeError("policy loading unavailable in bimanual-yam bridge")

        algo.algo_factory = _unavailable
        algo.RolloutPolicy = _unavailable
        sys.modules["robomimic.algo"] = algo


install()
