% A triaxial ellipsoid: the sphere with each axis scaled independently.
%
% Still degree-1 in the spherical harmonics — scaling a coordinate scales its
% coefficients — so the surface the solver sees is exactly the one written
% here, with nothing lost to band-limiting. The simplest geometry that is not
% the sphere, and the one whose curvature is easiest to reason about.

function [gx, gy, gz] = shape(theta, phi, ax, ay, az)
  st = sin(theta);
  gx = ax * (st .* cos(phi));
  gy = ay * (st .* sin(phi));
  gz = az * cos(theta);
end
