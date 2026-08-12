function dump_acoustics_test(input_matfile, output_matfile, coil)
%% dump_acoustics_test  Run check_grad_acoustics on a saved waveform and
%   dump its outputs, to cross-check ge/acoustics.py.

d = load(input_matfile);
grad = d.grad;
gdt = d.gdt;

[g_fresp, hz, val, esp_time, esp] = check_grad_acoustics(grad, coil, 0, gdt);

% flatten val (nested cell: {lg}{l1}{l2} = [magb rmsagb]) to a plain
% max-over-everything, matching ge/acoustics.py's simplified AcousticsResult
max_in_band = 0;
for lg = 1:length(val)
    for l1 = 1:length(val{lg})
        for l2 = 1:length(val{lg}{l1})
            max_in_band = max(max_in_band, val{lg}{l1}{l2}(1));
        end
    end
end

save(output_matfile, 'g_fresp', 'hz', 'max_in_band', '-v7.3');
fprintf('Wrote %s (max_in_band=%g)\n', output_matfile, max_in_band);
end
