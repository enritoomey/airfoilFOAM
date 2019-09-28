Include "perfil.geo";
Include "perfil_bl.geo";
Include "box.geo";

Spline(1) = {perfilStartpoint:perfilMidpoint};
Spline(2) = {perfilMidpoint:perfilEndpoint,perfilStartpoint};

Spline(3) = {blStartpoint:blMidpoint};
Spline(4) = {blMidpoint:blEndpoint};

Line(5) = {perfilMidpoint,blMidpoint};
Line(6) = {blStartpoint,perfilStartpoint};
Line(7) = {perfilStartpoint,blEndpoint};


Transfinite Line{1, 2, 3, 4} = 1.0 / lc Using Bump 4;
Transfinite Line{5,-6,7} = 10 Using Progression 1.2;

Line Loop(1) = {1,5,-3,6};
Line Loop(2) = {2,7,-4,-5};
Line Loop(4) = {3,4,-7, -6};

Plane Surface(1) = {1};
Transfinite Surface{1};
Recombine Surface{1};
extrados[] = Extrude {0, 0, 1} {
    Surface{1};
    Layers{1};
    Recombine;
};

Plane Surface(2) = {2};
Transfinite Surface{2};
Recombine Surface{2};
intrados[] = Extrude {0, 0, 1} {
    Surface{2};
    Layers{1};
    Recombine;
};


Plane Surface(3) = {3, 4};
ids[] = Extrude {0, 0, 1} {
    Surface{3};
    Layers{1};
    Recombine;
};

Physical Surface("inlet") = {ids[5]};
Physical Surface("topAndBottom") = {ids[{2,4}]};
Physical Surface("outlet") = {ids[3]};
Physical Volume("volume") = {extrados[1], intrados[1], ids[1]};
Physical Surface("frontAndBack") = {1,2, 3, ids[0], extrados[0], intrados[0]};
Physical Surface("airfoil") = {extrados[2], intrados[2]};