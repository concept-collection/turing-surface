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
  [U, V] = analys(us + noise, vs * ones(numel(noise), 1));
  [u, v] = synth(U, V);
end

function [Un, Vn, u, v] = step(U, V, lam, filt, gx, gy, gz, p1, p2, q2, r, jhat, a, b, D1, D2, dt, niter)
  % Grouped transforms -- [a, b] = synth(x, y) -- are explicit batching:
  % output k is the transform of input k, and the whole group runs as one
  % batched Legendre dispatch, or as many as the device's lane width allows
  % (src/mgpu/plan.ts, materializeTransforms). The grouping is a promise of
  % independence, never of a lane width, so the same source runs anywhere.
  [u, v] = synth(U, V);
  uuv = u .* u .* v;

  % Right-hand side of the implicit solve (I - dt*D*lap_g) Unew = B.
  ru = a - u + uuv;
  rv = b - uuv;
  [Ru, Rv] = analys(ru, rv);
  Bu = U + dt * Ru;
  Bv = V + dt * Rv;

  % Preconditioned solve, then iterate the geometric correction. jhat is
  % the host's minimax scale over the operator's symbol eigenvalues mu(x)
  % -- the inverse squared principal stretches of the embedding, direction
  % included (src/geom/geometry.ts, Jhat): preconditioning with lam/jhat
  % contracts every mode and direction at rate
  % (muMax - muMin)/(muMax + muMin) < 1 on any surface, where the plain
  % lam (jhat = 1) diverges wherever mu > 2 -- docs/reduced-transforms.md
  % Sec 10. The answer never depends on jhat (the lamJ term added inside
  % dLu is the term divided back out); only the convergence rate does. On
  % the sphere mu = 1 and lamJ = lam.
  lamJ = lam ./ jhat;
  Un = Bu ./ (1 + (dt * D1) * lamJ);
  Vn = Bv ./ (1 + (dt * D2) * lamJ);

  for k = 1:niter
    % dlap = lap_g - lap_s at the current iterate, in flux form
    % (docs/reduced-transforms.md Sec 4). The sin-weighted derivatives
    % sin(theta)*dtheta(u) and dphi(u) -- both smooth on the sphere,
    % synthesized straight from the dthetac/dphic coefficient shuffles --
    % are combined pointwise through the precomputed weights p1,p2,q2 into
    % two fluxes P,Q, also smooth. Their coefficients are then pushed
    % through the *same* shuffles again and summed before the one synthesis
    % of the divergence, which r scales into lap_g(u). The only division by
    % sin(theta) anywhere is folded into p1,p2,q2,r at precompute time.
    % lamJ.*Un adds back the preconditioner's -lap_s(Un)/jhat, since lam
    % holds +l(l+1). filt zeroes the top two degrees, where the derivative
    % recurrences cannot exactly represent a derivative -- and the correction
    % itself is projected onto the same band (algos.tex Algorithm 5 zeroes
    % the same coefficients): without that, each iteration replaces a bit
    % more of the top degrees' implicit diffusion with nothing (their fixed
    % point is the undiffused Bu), and the two species un-diffuse at
    % different rates -- a spurious Turing band at the band edge.
    %
    % The two species share each grouped call: the four gradient
    % syntheses, the four flux analyses, the two divergence syntheses and
    % the two final analyses each run as one batched dispatch.
    Fu = Un .* filt;
    Fv = Vn .* filt;
    vtu = dthetac(Fu);
    vpu = dphic(Fu);
    vtv = dthetac(Fv);
    vpv = dphic(Fv);
    [Ftu, Fpu, Ftv, Fpv] = synth(vtu, vpu, vtv, vpv);
    Pu = p1 .* Ftu + p2 .* Fpu;
    Qu = p2 .* Ftu + q2 .* Fpu;
    Pv = p1 .* Ftv + p2 .* Fpv;
    Qv = p2 .* Ftv + q2 .* Fpv;
    [PAu, QAu, PAv, QAv] = analys(Pu, Qu, Pv, Qv);
    Pcu = PAu .* filt;
    Qcu = QAu .* filt;
    Pcv = PAv .* filt;
    Qcv = QAv .* filt;
    scu = dthetac(Pcu) + dphic(Qcu);
    scv = dthetac(Pcv) + dphic(Qcv);
    [Lu, Lv] = synth(scu, scv);
    lapu = r .* Lu;
    lapv = r .* Lv;
    [LAu, LAv] = analys(lapu, lapv);
    dLu = (LAu + lamJ .* Un) .* filt;
    dLv = (LAv + lamJ .* Vn) .* filt;

    Un = (Bu + (dt * D1) * dLu) ./ (1 + (dt * D1) * lamJ);
    Vn = (Bv + (dt * D2) * dLv) ./ (1 + (dt * D2) * lamJ);
  end
end
