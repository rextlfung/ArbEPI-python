function dump_pns_test(matfile)
%% dump_pns_test  Run pge2.pns on the same synthetic waveform as pge2.pns.m's
%   own sub_test(), and dump inputs+outputs to a .mat file for cross-checking
%   ge/pns.py.

dt = 4e-6;   % gradient raster time, sec

% XRM coil (MR750) -- same coefficients as pge2.pns.m's sub_test()
chronaxie = 360e-6;
rheobase = 23.4;
alpha = 0.333;
s_min = rheobase / alpha;

dur = 1e-3;   % sec
ramp = linspace(0, 1, 100);
g = [ramp ones(1,200), fliplr(ramp)];
g = g(1:end-1);
g = repmat([g -g], [1 5]);
g = [g; 0.5*g; 0.5*g];
g = 4 * g * 1e-2;    % T/m

[pt, p] = pge2.pns(s_min, chronaxie, g, dt);

save(matfile, 'dt', 'chronaxie', 's_min', 'g', 'pt', 'p', '-v7.3');
fprintf('Wrote %s (max pt=%g)\n', matfile, max(pt));
end
