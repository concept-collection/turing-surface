% Schnakenberg reaction-diffusion on a closed surface.
%
%   du/dt = D1*lap_g(u) + a - u + u^2*v
%   dv/dt = D2*lap_g(v) + b     - u^2*v
%
% Explicit reaction, implicit diffusion (IMEX Euler). The implicit solve
% splits lap_g = lap_s + dlap: the round-sphere part lap_s is diagonal in
% spherical-harmonic space (eigenvalues -lam), and the loop iterates the
% geometric correction dlap from that exact solve. Grid fields are npts x 1;
% spectral fields are real 2 x nlm. See docs/richardson-iteration.md.
%
% The correction evaluates lap_g in flux form -- 6 transforms per species
% per iteration where the Cartesian-gradient form (Algorithm 4 of
% evolving_surface/notes/algos.tex) needs 12. See
% docs/reduced-transforms.md, and models/schnakenberg_alg4.m
% for the original form kept as a live reference.

function [U, V, u, v] = init(noise, a, b)
  us = a + b;
  vs = b / (us * us);
  U = analys(us + noise);
  V = analys(vs * ones(numel(noise), 1));
  u = synth(U);
  v = synth(V);
end

function [Un, Vn, u, v] = step(U, V, lam, filt, gx, gy, gz, p1, p2, q2, r, a, b, D1, D2, dt, niter)
  u = synth(U);
  v = synth(V);
  uuv = u .* u .* v;

  % Right-hand side of the implicit solve (I - dt*D*lap_g) Unew = B.
  Bu = U + dt * analys(a - u + uuv);
  Bv = V + dt * analys(b - uuv);

  % Round-sphere solve, then iterate the geometric correction.
  Un = Bu ./ (1 + (dt * D1) * lam);
  Vn = Bv ./ (1 + (dt * D2) * lam);

  for k = 1:niter
    % dlap = lap_g - lap_s at the current iterate, in flux form
    % (docs/reduced-transforms.md Sec 4). The sin-weighted
    % derivatives sin(theta)*dtheta(u) and dphi(u) -- both smooth on the
    % sphere, synthesized straight from the dthetac/dphic coefficient
    % shuffles -- are combined pointwise through the precomputed weights
    % p1,p2,q2 into two fluxes P,Q, also smooth. Their coefficients are then
    % pushed through the *same* shuffles again and summed before the one
    % synthesis of the divergence, which r scales into lap_g(u). The only
    % division by sin(theta) anywhere is folded into p1,p2,q2,r at precompute
    % time. lam.*Un adds back -lap_s(Un), since lam holds +l(l+1). filt
    % zeroes the top two degrees, where the derivative recurrences cannot
    % exactly represent a derivative.
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

    Fv = Vn .* filt;
    Ftv = synth(dthetac(Fv));
    Fpv = synth(dphic(Fv));
    Pv = p1 .* Ftv + p2 .* Fpv;
    Qv = p2 .* Ftv + q2 .* Fpv;
    Pcv = analys(Pv) .* filt;
    Qcv = analys(Qv) .* filt;
    scv = dthetac(Pcv) + dphic(Qcv);
    lapv = r .* synth(scv);
    dLv = analys(lapv) + lam .* Vn;

    Un = (Bu + (dt * D1) * dLu) ./ (1 + (dt * D1) * lam);
    Vn = (Bv + (dt * D2) * dLv) ./ (1 + (dt * D2) * lam);
  end
end
