Include "perfil.geo";
Include "box.geo";

Surface(1) = {3, 1};
Recombine Surface{1};

ids[] = Extrude {0, 0, 1}
{
    Surface{1};
    Layers{1};
    Recombine;
};

Physical Surface("outlet") = {ids[3]};
Physical Surface("topAndBottom") = {ids[{2, 4}]};
Physical Surface("inlet") = {ids[5]};
Physical Surface("airfoil") = {ids[{6:8}]};
Physical Surface("frontAndBack") = {ids[0], 1};
Physical Volume("volume") = {ids[1]};