"""Bridges one mirtorch LinearMap (recon/operators.py, recon/operators_b0.py)
into a pair of sigpy.linop.Linop objects, so sigpy's already-tested
regularization/solver machinery (recon/basic/recon_sigpy.py's Wavelet +
FiniteDifference + prox.Stack + PrimalDualHybridGradient pattern) can be
reused unchanged with this repo's own B0-corrected encoding operator in
place of sigpy.mri.linop.Sense -- rather than reimplementing PDHG, wavelet
transforms, or TV proximal operators from scratch in torch.

Only sigpy (pure Python + numpy) needs to exist in .venv-recon for this --
not cupy -- so each Linop.apply()/.adjoint() call round-trips its argument
through host memory: numpy -> torch (GPU) -> mirtorch forward/adjoint ->
numpy. This costs one small host<->device copy per call (a few MB for this
repo's per-frame image/k-space sizes), negligible next to the actual
FFT + smaps + (for the B0-corrected operator) L-segment loop compute it
wraps -- not the bottleneck, so not worth adding a cupy dependency to
avoid it.

Before reusing sigpy regularization/solver defaults (e.g.
recon_sigpy.py's lamb_l1/lamb_tv) with a torch_op bridged through here,
call recon/operators_b0.py's check_operator_unitary(torch_op, x0) once --
those defaults were tuned against sigpy.mri.linop.Sense, which is
genuinely unitary; a non-unitary torch_op (any B0-corrected
GatheredSenseB0, see that function's docstring) needs its own retuning,
not a transplanted lambda. This bridge class itself does not call it
automatically (a per-frame batch driver would otherwise pay a full power
iteration once per frame) -- callers building a new operator/driver
combination should check once, e.g. against frame 0's operator, before
looping.
"""

import numpy as np
import sigpy as sp
import torch
from mirtorch.linear.linearmaps import LinearMap


def _to_torch(x: np.ndarray, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=dtype)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


class TorchLinopBridge(sp.linop.Linop):
    """Forward half of the bridge: sigpy Linop wrapping torch_op.apply().

    torch_op: a mirtorch LinearMap (e.g. GatheredSenseB0) with size_in/
    size_out already fixed (one frame's operator, not the BlockDiagonal
    over every frame). device: where torch_op's own tensors (smaps,
    c_phasors, ...) already live -- inputs are moved there for the call
    and outputs are always returned to host (numpy), matching every other
    sigpy Linop's convention of operating on whatever array it's given.
    """

    def __init__(self, torch_op: LinearMap, device: torch.device):
        self.torch_op = torch_op
        self.device = device
        super().__init__(list(torch_op.size_out), list(torch_op.size_in))

    def _apply(self, input: np.ndarray) -> np.ndarray:
        x = _to_torch(input, torch.complex64, self.device)
        y = self.torch_op.apply(x)
        return _to_numpy(y)

    def _adjoint_linop(self):
        return TorchLinopBridgeAdjoint(self.torch_op, self.device, fwd=self)


class TorchLinopBridgeAdjoint(sp.linop.Linop):
    """Adjoint half: sigpy Linop wrapping torch_op.adjoint(). Never
    constructed directly -- returned by TorchLinopBridge._adjoint_linop()
    (and vice versa via its own _adjoint_linop() below), matching sigpy's
    own FFT/IFFT pairing convention (recon/sigpy_b0/sigpy_torch_bridge.py's module
    docstring)."""

    def __init__(self, torch_op: LinearMap, device: torch.device, fwd: TorchLinopBridge):
        self.torch_op = torch_op
        self.device = device
        self._fwd = fwd
        super().__init__(list(torch_op.size_in), list(torch_op.size_out))

    def _apply(self, input: np.ndarray) -> np.ndarray:
        y = _to_torch(input, torch.complex64, self.device)
        x = self.torch_op.adjoint(y)
        return _to_numpy(x)

    def _adjoint_linop(self):
        return self._fwd
