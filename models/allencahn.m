% Allen-Cahn on a closed surface: interfaces form, then coarsen.
%
%   du/dt = eps2*lap_g(u) + u - u^3
%
% Same scheme as models/schnakenberg.m.

function [U, u] = init(noise)
  U = analys(noise);
  u = synth(U);
end

function [Un, u] = step(U, lam, filt, gx, gy, gz, p1, p2, q2, r, eps2, dt, niter)
  u = synth(U);

  Bu = U + dt * analys(u - u.^3);
  Un = Bu ./ (1 + (dt * eps2) * lam);

  for k = 1:niter
    % dlap = lap_g - lap_s, evaluated at the current iterate in flux form
    % (see models/schnakenberg.m, docs/richardson-iteration.md and
    % docs/reduced-transforms.md for the derivation).
    Fu = Un .* filt;
    Ftu = synth(dthetac(Fu));
    Fpu = synth(dphic(Fu));
    Pu = p1 .* Ftu + p2 .* Fpu;
    Qu = p2 .* Ftu + q2 .* Fpu;
    Pcu = analys(Pu) .* filt;
    Qcu = analys(Qu) .* filt;
    scu = dthetac(Pcu) + dphic(Qcu);
    lapu = r .* synth(scu);
    dLu = analys(lapu) + lam .* Un;

    Un = (Bu + (dt * eps2) * dLu) ./ (1 + (dt * eps2) * lam);
  end
end
