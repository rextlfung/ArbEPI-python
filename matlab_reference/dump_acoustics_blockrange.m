function dump_acoustics_blockrange(seqfile, coil, matfile)
%% dump_acoustics_blockrange  Reproduce write_to_ge_from_seq.m's exact
%   acoustics check (pge2.plot with blockRange [1 10], then
%   check_grad_acoustics) and dump the actual waveform + result, to
%   reconcile against ge/acoustics.py run on the SAME waveform.

sysGE = pge2.opts(150e-6, 120e-6, 0.25, 10, 20, coil);
ceq = seq2ceq(seqfile);

hfig = figure('Visible', 'off');
S = pge2.plot(ceq, sysGE, 'blockRange', [1 10], 'rotate', false, 'interpolate', true, 'wt', [0.8 1.0 0.7]);
close(hfig);

grad = reshape([S.gx.signal S.gy.signal S.gz.signal], [length(S.gx.signal), 1, 3])/100;
[g_fresp, hz, val, esp_time, esp] = check_grad_acoustics(grad, coil, 0);

max_in_band = 0;
for lg = 1:length(val)
    for l1 = 1:length(val{lg})
        for l2 = 1:length(val{lg}{l1})
            max_in_band = max(max_in_band, val{lg}{l1}{l2}(1));
        end
    end
end

save(matfile, 'grad', 'max_in_band', '-v7.3');
fprintf('%s: n_samples=%d max_in_band=%g\n', seqfile, size(grad,1), max_in_band);
end
