B = [1,-0.7653668, 0.99999];
A = [1, -0.722744, 0.888622];
fvtool(B, A, 'Analysis','polezero')
z = roots(B);
p = roots(A);
results = abs(p);
h = impz(B,A);
