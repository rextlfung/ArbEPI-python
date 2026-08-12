function dump_pns_peak(seqfile, sysPGE2_coil, matfile)
%% dump_pns_peak  Compute peak PNS%% across EVERY segment instance in a real
%   sequence, using the actual per-instance MATLAB pipeline
%   (seq2ceq -> getsegmentinstance -> pge2.pns), but WITHOUT checksegment.m's
%   fail-fast throw on >100%% -- so the true peak is captured for comparison
%   against ge/check.py's whole-sequence-convolution simplification, instead
%   of just learning that it exceeded 100%% somewhere.
%
%   sysPGE2_coil: coil code, e.g. 'hrmbuhp' (see pge2.opts.m)

switch lower(sysPGE2_coil)
    case 'xrm',      chronaxie=334e-6; rheobase=23.4; alpha=0.333;
    case 'hrmbuhp',  chronaxie=359e-6; rheobase=26.5; alpha=0.370;
    otherwise, error('add this coil''s coefficients from pge2.opts.m');
end

ceq = seq2ceq(seqfile);

% GE_UHP hardware constants, matching scanners.py's SCANNERS['GE_UHP']
sysGE = pge2.opts(150e-6, 120e-6, 0.25, 10, 20, sysPGE2_coil);
GRAD_UPDATE_TIME = sysGE.GRAD_UPDATE_TIME;

Smin = rheobase / alpha;
peak_pns = 0;
peak_row = 0;

n = 1;
textprogressbar('dump_pns_peak: scanning segment instances: ');
while n < ceq.nMax
    i = ceq.loop(n,1);
    L = ceq.loop(n:(n-1+ceq.segments(i).nBlocksInSegment), :);
    S = pge2.getsegmentinstance(ceq, i, sysGE, L, 'rotate', true, 'interpolate', true);

    G = [S.gx.signal'; S.gy.signal'; S.gz.signal']/100;  % T/m
    if ~isempty(G) && size(G,2) > 1
        [pt, ~] = pge2.pns(Smin, chronaxie, G, GRAD_UPDATE_TIME, 'wt', [0.8 1.0 0.7]);
        [m, idx] = max(pt);
        if m > peak_pns
            peak_pns = m;
            peak_row = n;
        end
    end

    textprogressbar(n/ceq.nMax*100);
    n = n + ceq.segments(i).nBlocksInSegment;
end
textprogressbar(' done');

save(matfile, 'peak_pns', 'peak_row', '-v7.3');
fprintf('\nWrote %s (peak_pns=%.2f%% at row %d)\n', matfile, peak_pns, peak_row);
end
