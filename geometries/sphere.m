% The unit sphere.
%
% A geometry file defines one function, `shape`, giving the Cartesian
% coordinates of the surface over the solver's (theta, phi) grid. Both inputs
% are npts x 1 grid fields, as are all three outputs, and every line is
% element-wise MATLAB compiled to a WebGPU kernel — the same backend the models
% go through.
%
% The host then analyses the result into spherical-harmonic coefficients, which
% is the form the geometry is carried in: band-limited at lmax, evaluable on
% any grid. For this file that costs nothing, because x, y and z ARE degree-1
% harmonics — which is what makes this the case where everything downstream
% reduces exactly to the round-sphere solver.

function [gx, gy, gz] = shape(theta, phi)
  st = sin(theta);
  gx = st .* cos(phi);
  gy = st .* sin(phi);
  gz = cos(theta);
end
