% The unit sphere — the reference case.
%
% A geometry file defines shape(theta, phi, ...) -> gx, gy, gz: the surface
% over the solver's grid (all npts x 1). Unlike the models it runs once, on
% the CPU through numbl's interpreter, so the full MATLAB subset is
% available — loops, arrays, min/max, legendre, seeded randomness via
% rng/randn. The host analyses the result into spherical-harmonic
% coefficients, band-limited at lmax.

function [gx, gy, gz] = shape(theta, phi)
  st = sin(theta);
  gx = st .* cos(phi);
  gy = st .* sin(phi);
  gz = cos(theta);
end
