#=
b0map.jl -- B0 field map from the dual-echo deGRE, with MRIFieldmaps.jl's
`b0map` (C Y Lin, J A Fessler, "Efficient Regularized Field Map Estimation in
3D MRI", IEEE TCI 2020; https://github.com/MagneticResonanceImaging/MRIFieldmaps.jl).

Called by preprocess/b0map.py's estimate_b0map, which writes a temporary h5
with `ksp_gre_echoes` [Nx, Ny, Nz, n_echoes, Ncoils] (whitened k-space), a
`TE_degre` attribute (s), and optionally `smaps_degre`/`emap_degre`.

    julia --project=preprocess/julia preprocess/julia/b0map.jl \
        <gre_h5> <output_h5> [smaps_h5] [eig_mask_threshold] [mask_threshold] [precon]

Writes `b0map_hz`, `finit_hz` and `mask` on the deGRE grid.

Choices:
- precon = :diag, not MRIFieldmaps' default :ichol. With :ichol the
  preconditioner is built from the same roughness operator as the gradient, so
  the regularization weight l2b had almost no effect (roughness flat for l2b in
  [-6, 28]) and the map was speckled; :diag gave a ~4x smoother map and cut
  B0-correction speckle in reconstructions ~3x. l2b/niter stay at the library
  defaults (-6, 30).
- The fit mask is mandatory: first-echo magnitude above mask_threshold x peak
  (MRIFieldmaps' b0init default 0.1), ANDed with emap > eig_mask_threshold when
  maps are given. Without a mask, an exactly-zero voxel gives 0/0 in the coil
  combine and the NaN spreads to the whole map.
- With sensitivity maps, coils are combined with them (matched filter) instead
  of MRIFieldmaps' phase-contrast fallback, which weights each coil by its own
  noisy first-echo image.
- The starting point finit is ROMEO-unwrapped (Dymerska et al., MRM 2021) phase
  difference / (2 pi dTE). b0map's NCG only finds the local optimum near its
  start, so a wrapped start keeps the aliasing: on a +-450 Hz synthetic field
  (dTE 2 ms, naive range +-250 Hz) the error was 207 Hz RMSE wrapped vs <60 Hz
  unwrapped. The unwrapped array is the phase-contrast combine's echo 2,
  y2 conj(y1)/sos (its echo 1 is identically real), weighted by its magnitude.
- Axis order: HDF5.jl reads h5py-written arrays with axes reversed, so arrays
  are permuted with reverse(1:ndims) on read and on write; the output is in
  numpy axis order.
=#

using FFTW: ifft, fftshift, ifftshift
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
ifftshift3(x) = ifftshift(x, (1, 2, 3))

"Centered inverse 3D FFT, fftshift(ifft(ifftshift(.))) -- the same pairing as
preprocess/utils.py's `ift3c`."
ifft3c(x) = fftshift3(ifft(ifftshift3(x), (1, 2, 3)))

function load_gre_images(gre_h5_path::AbstractString)
    ksp, TE = h5open(gre_h5_path, "r") do f
        haskey(f, "ksp_gre_echoes") ||
            error("b0map.jl: '$gre_h5_path' has no 'ksp_gre_echoes' dataset.")
        ksp = read_numpy_array(f, "ksp_gre_echoes")  # (Nx, Ny, Nz, n_echoes, Ncoils)
        haskey(attributes(f), "TE_degre") ||
            error("b0map.jl: '$gre_h5_path' has no 'TE_degre' attribute -- " *
                  "the acquisition's scan_info.mat predates the dual-echo deGRE.")
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
    eig_mask_threshold::Real = 0.95,
    threshold::Real = 0.1, precon::Symbol = :diag,
)
    println("Loading '$gre_h5_path'...")
    images, echotime = load_gre_images(gre_h5_path)
    println("  images size (Nx, Ny, Nz, Ncoils, Nechoes): ", size(images))
    println("  TE_degre (s): ", echotime)

    mask = magnitude_mask(images, threshold)
    println("  magnitude mask: $(count(mask)) / $(length(mask)) voxels above $(threshold) x peak magnitude")

    # Optional: real sensitivity maps (preprocess/smaps.py's ESPIRiT
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
