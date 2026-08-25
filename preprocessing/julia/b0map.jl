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
        <gre_h5_path> <output_h5_path> [mask_threshold]

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

Coil sensitivity maps (`smap`) are deliberately not passed: this pipeline's
existing ESPIRiT maps (smaps.py) are estimated on a `cal_size`-cropped grid
(24^3 by default) and then resized to the *EPI* acquisition grid --
neither matches the deGRE grid this script's `images` array is defined on
(Nx_degre, Ny_degre, Nz_degre), so passing them without a matching resize
step would silently violate `b0map`'s `smap` shape check. MRIFieldmaps
falls back to its own phase-contrast coil combine (Bernstein et al., MRM
1994, eqn 13) when `smap` is omitted, which needs no such alignment.

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

function main(gre_h5_path::AbstractString, output_h5_path::AbstractString, threshold::Real = 0.1)
    println("Loading '$gre_h5_path'...")
    images, echotime = load_gre_images(gre_h5_path)
    println("  images size (Nx, Ny, Nz, Ncoils, Nechoes): ", size(images))
    println("  TE_degre (s): ", echotime)

    mask = magnitude_mask(images, threshold)
    println("  mask: $(count(mask)) / $(length(mask)) voxels above $(threshold) x peak magnitude")

    println("Unwrapping finit via ROMEO...")
    finit = romeo_finit(images, echotime, mask)
    println("  finit range (Hz, masked): ", extrema(finit[mask]))

    println("Running MRIFieldmaps.b0map...")
    fhat, _times, _out = b0map(finit, images, echotime; mask, chat = true)

    mkpath(dirname(output_h5_path))
    h5open(output_h5_path, "w") do f
        write_numpy_array(f, "b0map_hz", Float32.(fhat))
        write_numpy_array(f, "finit_hz", Float32.(finit))
        write_numpy_array(f, "mask", Array{Bool}(mask))
        f["TE_degre"] = collect(echotime)
        attributes(f)["mask_threshold"] = threshold
    end
    println("Wrote '$output_h5_path'.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in (2, 3) ||
        error("usage: julia b0map.jl <gre_h5_path> <output_h5_path> [mask_threshold]")
    args = length(ARGS) == 3 ? (ARGS[1], ARGS[2], parse(Float64, ARGS[3])) : (ARGS[1], ARGS[2])
    main(args...)
end
