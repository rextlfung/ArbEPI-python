"""Block-design BOLD activation in a region placed in world coordinates.

SNAKE's BlockActivationHandler takes its ROI either from a Harvard-Oxford atlas
label (downloaded by nilearn) or, with atlas=None, from an ellipsoid in the
occipital cortex. That ellipsoid is defined in voxels of BrainWeb's full
0.5 mm grid and scaled by the ratio of array shapes, so it lands in the right
place only on a phantom that still spans BrainWeb's whole field of view. The
engine applies handlers after resampling the phantom to the acquisition grid,
which for ArbEPI is a different field of view.

EllipsoidActivationHandler places the same ellipsoid in millimeters through the
phantom's affine instead, so the ROI is the same piece of anatomy on any grid.
The BOLD model is SNAKE's unchanged: the ROI is a copy of gray matter whose
weight follows the block design convolved with the HRF, peaking at a fractional
signal change of TE / delta_r2s (TE in ms).

One default differs: oversampling is 1, not 50. It is the factor by which SNAKE
refines the time grid, already one point per excitation, before convolving the
stimulus with the HRF, and the convolution's cost grows with its square. At a
50 ms shot TR, 50 means a 1 ms grid and minutes to hours for one regressor; 1
changes the regressor by about 1% of its peak (a shift of half a shot TR).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from snake.core.handlers import BlockActivationHandler
from snake.core.phantom import Phantom

# snake.core.handlers.activations.roi.BRAINWEB_OCCIPITAL_ROI (center (185, 52,
# 145) and semi-axes (100, 20, 50) voxels of the 0.5 mm grid whose first voxel
# is at (-90.25, -126.25, -72.25) mm), converted to mm.
OCCIPITAL_CENTER_MM = (2.25, -100.25, 0.25)
OCCIPITAL_SEMI_AXES_MM = (50.0, 10.0, 25.0)
OCCIPITAL_EULER_ANGLES = (0.0, 0.0, -5.0)

# The hand area of one hemisphere's primary motor cortex, at its usual place in
# the stereotaxic frame BrainWeb's models are in: around (-38, -22, 56) mm, on
# the central sulcus near the vertex. An ellipsoid holding 3.9 cm3 of subject
# 4's gray matter there, not an anatomical parcellation; which hemisphere the
# negative x is has not been checked against the phantom.
MOTOR_CENTER_MM = (-38.0, -22.0, 56.0)
MOTOR_SEMI_AXES_MM = (14.0, 12.0, 14.0)
MOTOR_EULER_ANGLES = (0.0, 0.0, 0.0)


def ellipsoid_mask(
    shape: tuple[int, int, int],
    affine: NDArray,
    center_mm: tuple[float, float, float],
    semi_axes_mm: tuple[float, float, float],
    euler_angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> NDArray[np.bool_]:
    """Voxels of a (shape, affine) grid whose centers lie inside an ellipsoid
    given in world mm. euler_angles: 'zyx' degrees, as in SNAKE's
    get_indices_inside_ellipsoid."""
    idx = np.indices(shape).reshape(3, -1).T
    world = idx @ np.asarray(affine)[:3, :3].T + np.asarray(affine)[:3, 3]
    local = Rotation.from_euler('zyx', euler_angles, degrees=True).apply(world - center_mm)
    return (np.linalg.norm(local / np.asarray(semi_axes_mm), axis=1) <= 1).reshape(shape)


class EllipsoidActivationHandler(BlockActivationHandler):
    """Block-design activation in (base tissue) x (an ellipsoid given in mm).

    Takes BlockActivationHandler's parameters (block_on, block_off, duration,
    offset, delta_r2s, hrf_model, base_tissue_name, roi_threshold, ...) plus
    the ellipsoid: center_mm, semi_axes_mm, euler_angles. The defaults are
    SNAKE's occipital-cortex ROI for BrainWeb.
    """

    __handler_name__ = 'activation-block-ellipsoid'

    center_mm: tuple[float, float, float] = OCCIPITAL_CENTER_MM
    semi_axes_mm: tuple[float, float, float] = OCCIPITAL_SEMI_AXES_MM
    euler_angles: tuple[float, float, float] = OCCIPITAL_EULER_ANGLES
    atlas: str | None = None
    oversampling: int = 1

    def _get_roi_base(self, phantom: Phantom) -> NDArray:
        if self.base_tissue_name not in phantom.labels_idx:
            raise ValueError(f'Tissue {self.base_tissue_name} not found in the phantom.')
        tissue = phantom.masks[phantom.labels_idx[self.base_tissue_name]]
        inside = ellipsoid_mask(
            tissue.shape, phantom.affine, self.center_mm, self.semi_axes_mm, self.euler_angles
        )
        return np.where(inside, tissue, 0).astype(tissue.dtype)
