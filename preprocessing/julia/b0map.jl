#=
b0map.jl -- B0 field map estimation from preprocess.py's dual-echo GRE cache.

Consumes `<seqname>_gre.h5` (written by preprocessing/preprocess.py STEP 2:
whitened + coil-compressed k-space, every deGRE echo, plus a `TE_degre`
attribute in seconds) and writes `<seqname>_b0map.h5` (a regularized B0
field map in Hz, via MRIFieldmaps.jl's `b0map`, ../MRIFieldmaps.jl reference
https://github.com/MagneticResonanceImaging/MRIFieldmaps.jl,
algorithm: C Y Lin, J A Fessler, "Efficient Regularized Field Map
Estimation in 3D MRI", IEEE TCI 2020).

Usage:
    julia --project=preprocessing/julia preprocessing/julia/b0map.jl \
        <gre_h5_path> <output_h5_path> [smaps_h5_path] [eig_mask_threshold] \
        [mask_threshold] [precon]

`precon` (default `:diag`, overriding `b0map`'s own default `:ichol`) is
its NCG preconditioner -- the one parameter actually worth overriding here.
`l2b`/`niter` (its regularization weight and NCG iteration count) are
deliberately *not* exposed, despite an earlier investigation adding and
sweeping both: `l2b` turned out to have essentially no effect on the fitted
field map's smoothness under `:ichol`, regardless of `niter` -- traced to
`:ichol`'s preconditioner (`H = spdiagm(hcurv) + CC`) being built from the
*same* `CC = beta * C'C` roughness-penalty operator that appears in the
gradient (`grad = hderiv + CC*w`): as `beta` grows, both numerator and
denominator become CC-dominated and the preconditioned step `H^-1 * grad`
collapses toward `-w` independent of `beta` (confirmed empirically:
mean|Laplacian| roughness flat at ~8.2-8.4 for `l2b` in [-6, 28], a
16384x-to-270-million-x range in beta, at both niter=30 and a fully
2000-iteration-converged run). Switching `precon` to `:diag` instead fixes
this directly -- `:diag`'s own preconditioner (`Hdiag = hcurv + diag(CC)`)
doesn't have the same numerator/denominator cancellation, since it only
uses `CC`'s diagonal, not the full matrix -- and gets a substantially
smoother field map (roughness ~4x lower than `:ichol`'s at MRIFieldmaps'
own `l2b`/`niter` defaults, -6.0/30, visibly confirmed in a real
reconstruction: B0-corrected image roughness dropped from +61 excess over
an uncorrected baseline to +19.5 -- see git history / session notes for the
comparison figures). With `precon=:diag` fixing the actual problem, `l2b`
reverts to being a non-load-bearing knob not worth the CLI surface area of
exposing.

Independent corroboration from MRIFieldmaps.jl's own maintainer: its
[`02-b0map.jl` example/docs](https://github.com/MagneticResonanceImaging/MRIFieldmaps.jl/blob/main/docs/lit/examples/02-b0map.jl)
compares `:I`/`:diag`/`:chol`/`:ichol` on its own canonical test case and
closes with "it is interesting that in this Julia implementation the
diagonal preconditioner seems to be as effective as the incomplete
Cholesky preconditioner" -- i.e. this isn't specific to our data/problem,
it's a known characteristic of this package's Julia port (unlike the
original MATLAB implementation, where `ichol` is the expected/reliable
win). That upstream comparison is about final RMSE-vs-wall-time, though,
not about `l2b` sensitivity specifically -- it doesn't independently
confirm the numerator/denominator-cancellation mechanism above, which was
traced directly in this pipeline's own code.

`mask_threshold` (default 0.1, matching MRIFieldmaps' own `b0init` default)
sets the fraction of peak first-echo magnitude below which a voxel is
excluded from the fit. An explicit mask is *not* optional here (unlike
b0map's own `mask` keyword, which defaults to `trues(size(finit))`):
MRIFieldmaps' no-smap coil combine (`coil_combine`) divides by each voxel's
sum-of-squares coil magnitude, and any voxel whose magnitude is *exactly*
zero -- confirmed empirically with a synthetic all-zero-background test
volume -- produces 0/0 = NaN there; Julia's `maximum` propagates a single
NaN through the whole array, which zeroes out the "good pixel" background
threshold inside `b0init` and returns an all-NaN field map. Real scanner
data never has an exactly-zero voxel (thermal noise), so this wouldn't
reproduce in practice, but masking out background explicitly is the
standard/expected use of this package regardless (its own README example
and `b0init` both build a `finit` this same way) and removes any
sensitivity to this edge case.

Coil sensitivity maps (`smap`) are now passed, when `smaps_h5_path` is
given: `preprocessing/smaps.py`'s `load_smaps` resizes the same
`cal_size`-cropped ESPIRiT calibration (`smaps_raw`/`emap`) it already
produces for the *EPI* grid onto *this* deGRE grid too (`smaps_degre`,
`emap_degre`, in `<datdir>/recon/smaps_<seqname>_sigpy.h5`), so `smap`'s
shape now matches `images`' spatial dims exactly, satisfying `b0map`'s
`smap` shape check. This replaces MRIFieldmaps' own phase-contrast
coil-combine fallback (`coil_combine(images, nothing)`, Bernstein et al.,
MRM 1994, eqn 13) with a true matched-filter combine
(`coil_combine(images, smap)`, same reference, eqn 13's `smap`-provided
branch) -- the fallback weights each coil by its own noisy first-echo
image (`y1`) rather than a smooth sensitivity estimate, so its combine
noise directly reflects each coil's raw per-voxel SNR, worst exactly in
this pipeline's low-per-coil-SNR object-center regions (see the `l2b`
paragraph above). `emap_degre` (ESPIRiT's dominant-eigenvalue map, also
resized to this grid) is thresholded the same way `smaps.py`'s own
`eig_mask` is and ANDed into `mask` -- a cheap, ESPIRiT-informed
complement to the magnitude-only mask below, since it's already being
loaded here regardless. When `smaps_h5_path` is empty (default), behavior
is unchanged from before: no `smap`, `mask` is magnitude-only.

`finit` (the NCG solve's starting point) is built here via
[ROMEO.jl](https://github.com/korbinian90/ROMEO.jl) (Dymerska et al.,
"Phase unwrapping with a rapid opensource minimum spanning tree algorithm
(ROMEO)", MRM 2021) rather than left to `b0map`'s own default. Passed a
plain two-point phase difference, `b0map`'s NCG solve can converge to the
wrong 2π branch wherever `finit` itself is wrong by a full cycle -- its
data-fit term is periodic (see the module-level algorithm note below), but
that only guarantees a *locally* consistent optimum, not that NCG finds
its way to the globally correct branch from a badly-aliased start. This
was measurable, not theoretical: on a synthetic field map exceeding the
default `finit`'s +-1/(2 dTE) unambiguous range (dTE = 2 ms => +-250 Hz),
`b0map` fed the unwrapped `finit` recovered the true field to 34 Hz RMSE;
fed the plain wrapped `finit`, 207 Hz RMSE -- it converged to a local
minimum that reproduced the aliasing instead of correcting it.

Which array to unwrap is *not* `angle.(zdata[...,1])` -- `coil_combine`'s
phase-contrast formula (`zdata_e = sum_c conj(y_{c,1}/sos) * y_{c,e}`)
makes the reference echo's own combined phase identically zero by
construction (`conj(y_{c,1}) * y_{c,1} / sos` sums to a real positive
number for every voxel), confirmed empirically -- a first attempt at
spatially unwrapping `zdata[...,1]` changed exactly zero voxels, on data
that plainly needed it. The physically meaningful signal is
`zdata[...,2]`: for this two-echo case it reduces to exactly `y2 * conj(y1)
/ sos`, i.e. the same wrapped phase *difference* `b0init` itself computes
(`angle.(y2 .* conj(y1))`) -- so it's this one 3D volume that gets handed
to `ROMEO.unwrap`, and its own magnitude (a coherence-like quantity in
[0, 1] from the phase-contrast combine, not the raw image amplitude --
naturally lower wherever the two echoes disagree, i.e. exactly where
`unwrap` should trust the local phase less) is what weights it. Only the
first two echoes are used, matching `MRIFieldmaps.b0init`'s own
restriction to two-point phase difference in the non-water-fat case --
consistent with this pipeline only ever acquiring a two-echo deGRE.

Axis order, both directions: HDF5.jl stores/reads arrays reversed relative
to h5py/numpy (row-major C convention on disk vs. Julia's column-major
convention in memory) -- verified empirically against a real preprocess.py
GRE cache: a Python-written `(Nx, Ny, Nz, n_echoes, Ncoils)` dataset comes
back from `read` as `(Ncoils, n_echoes, Nz, Ny, Nx)`, and
`permutedims(raw, reverse(1:ndims(raw)))` recovers the correct array (same
correction preprocessing/matio.py documents and applies in the opposite
direction, for hdf5storage-written files read back by h5py). The reverse
permutedims is applied here on *write* too, so this script's own output
lands on disk already in numpy axis order and needs no correction from the
Python side (matching every other h5 file this pipeline writes for its own
use -- see preprocessing/config.py's `.h5`-vs-`.mat` convention note).
=#

using FFTW: ifft, fftshift
using HDF5: h5open, attributes
using MRIFieldmaps: b0map, coil_combine
using ROMEO: unwrap

function read_numpy_array(file, name::AbstractString)
    raw = read(file, name)
    permutedims(raw, reverse(1:ndims(raw)))
end

write_numpy_array(file, name::AbstractString, arr) =
    file[name] = permutedims(arr, reverse(1:ndims(arr)))

fftshift3(x) = fftshift(x, (1, 2, 3))

"Centered inverse 3D FFT -- same fftshift(ifft(fftshift(.))) convention as
preprocessing/run_rss.py's `_ift3` (applied here per echo/coil to bring
each fully-sampled Cartesian deGRE k-space volume to image space)."
ifft3c(x) = fftshift3(ifft(fftshift3(x), (1, 2, 3)))

function load_gre_images(gre_h5_path::AbstractString)
    ksp, TE = h5open(gre_h5_path, "r") do f
        haskey(f, "ksp_gre_echoes") ||
            error("b0map.jl: '$gre_h5_path' has no 'ksp_gre_echoes' dataset -- " *
                  "was it written by a params.mat snapshot without TE_degre (pre-dual-echo deGRE)?")
        ksp = read_numpy_array(f, "ksp_gre_echoes")  # (Nx, Ny, Nz, n_echoes, Ncoils)
        haskey(attributes(f), "TE_degre") ||
            error("b0map.jl: '$gre_h5_path' has no 'TE_degre' attribute -- " *
                  "regenerate params.mat via the current sequences/ArbEPI.py before preprocessing.")
        TE = read(attributes(f)["TE_degre"])
        (ksp, TE)
    end

    size(ksp, 4) >= 2 ||
        error("b0map.jl: need >= 2 echoes for field map estimation, got $(size(ksp, 4)).")

    img = similar(ksp, ComplexF32)
    for e in axes(ksp, 4), c in axes(ksp, 5)
        img[:, :, :, e, c] = ifft3c(ksp[:, :, :, e, c])
    end
    # MRIFieldmaps wants (dims..., nc, ne); preprocess.py's cache is (dims..., ne, nc).
    images = permutedims(img, (1, 2, 3, 5, 4))
    (images, Float32.(TE))
end

function magnitude_mask(images, threshold::Real)
    sos1 = dropdims(sqrt.(sum(abs2, images[:, :, :, :, 1]; dims = 4)); dims = 4)
    sos1 .> (threshold * maximum(sos1))
end

"ROMEO-unwrapped field map initial guess, in Hz -- see the module docstring
for why `zdata[...,2]` (not `zdata[...,1]`, which is identically zero) is
the array that actually needs unwrapping."
function romeo_finit(images, echotime, mask)
    zdata, _sos = coil_combine(images, nothing)  # (Nx, Ny, Nz, ne)
    dphi_wrapped = Float32.(angle.(zdata[:, :, :, 2]))
    dphi_mag = Float32.(abs.(zdata[:, :, :, 2]))
    dphi_unwrapped = unwrap(dphi_wrapped; mag = dphi_mag, mask = mask)
    dphi_unwrapped ./ Float32(2π * (echotime[2] - echotime[1]))
end

function main(
    gre_h5_path::AbstractString, output_h5_path::AbstractString,
    smaps_h5_path::AbstractString = "",
    eig_mask_threshold::Real = 0.2,
    threshold::Real = 0.1, precon::Symbol = :diag,
)
    println("Loading '$gre_h5_path'...")
    images, echotime = load_gre_images(gre_h5_path)
    println("  images size (Nx, Ny, Nz, Ncoils, Nechoes): ", size(images))
    println("  TE_degre (s): ", echotime)

    mask = magnitude_mask(images, threshold)
    println("  magnitude mask: $(count(mask)) / $(length(mask)) voxels above $(threshold) x peak magnitude")

    # Optional: real sensitivity maps (preprocessing/smaps.py's ESPIRiT
    # calibration, resized to this deGRE grid) in place of MRIFieldmaps'
    # phase-contrast coil-combine fallback -- see module docstring. Its
    # eigenvalue map is also used to tighten the magnitude-based mask
    # (same threshold convention as smaps.py's process_smaps' own
    # eig_mask), a cheap addition once smap is already being loaded here;
    # ROMEO unwrapping (below) uses this combined mask too, not just b0map.
    smap = nothing
    if !isempty(smaps_h5_path)
        println("Loading sensitivity maps from '$smaps_h5_path'...")
        h5open(smaps_h5_path, "r") do f
            smap = read_numpy_array(f, "smaps_degre")
            eig_mask = read_numpy_array(f, "emap_degre") .> eig_mask_threshold
            mask = mask .& eig_mask
        end
        println("  combined (magnitude & ESPIRiT-eigenvalue) mask: " *
                "$(count(mask)) / $(length(mask)) voxels")
    end

    println("Unwrapping finit via ROMEO...")
    finit = romeo_finit(images, echotime, mask)
    println("  finit range (Hz, masked): ", extrema(finit[mask]))

    # l2b/niter deliberately not passed -- MRIFieldmaps' own defaults
    # (-6.0/30) are used implicitly; see module docstring's `precon`
    # paragraph for why they're not worth exposing/overriding here.
    println("Running MRIFieldmaps.b0map (precon=$precon, " *
            "smap=$(isnothing(smap) ? "none" : "provided"))...")
    fhat, _times, _out = b0map(finit, images, echotime; smap, mask, precon)

    mkpath(dirname(output_h5_path))
    h5open(output_h5_path, "w") do f
        write_numpy_array(f, "b0map_hz", Float32.(fhat))
        write_numpy_array(f, "finit_hz", Float32.(finit))
        write_numpy_array(f, "mask", Array{Bool}(mask))
        f["TE_degre"] = collect(echotime)
        attributes(f)["mask_threshold"] = threshold
        attributes(f)["precon"] = String(precon)
        attributes(f)["used_smap"] = !isnothing(smap)
        attributes(f)["eig_mask_threshold"] = eig_mask_threshold
    end
    println("Wrote '$output_h5_path'.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    2 <= length(ARGS) <= 6 ||
        error("usage: julia b0map.jl <gre_h5_path> <output_h5_path> [smaps_h5_path] " *
              "[eig_mask_threshold] [mask_threshold] [precon]")
    args = (ARGS[1], ARGS[2],
            (length(ARGS) >= 3 ? (ARGS[3],) : ())...,
            (length(ARGS) >= 4 ? (parse(Float64, ARGS[4]),) : ())...,
            (length(ARGS) >= 5 ? (parse(Float64, ARGS[5]),) : ())...,
            (length(ARGS) >= 6 ? (Symbol(ARGS[6]),) : ())...)
    main(args...)
end
