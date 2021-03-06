// Autogenerated from examples/primal_introduction_ex

// To turn this Asymptote source file into an image for inclusion in
// Axom's documentation,
// 1. run Asymptote:
//    asy -f png showOrientation.asy
// 2. Optionally, use ImageMagick to convert the white background to transparent:
//    convert showOrientation.png -transparent white showOrientation.png

// preamble
settings.render = 6;
import three;
size(6cm, 0);

// axes
draw(O -- 1.7X, arrow=Arrow3(DefaultHead2), L=Label("$x$", position=EndPoint));
draw(O -- 2.4Y, arrow=Arrow3(), L=Label("$y$", position=EndPoint));
draw(O -- 2Z, arrow=Arrow3(), L=Label("$z$", position=EndPoint, align=W));

// triangle
path3 tri = (1.2,0,0)--(0,1.8,0)--(0,0,1.4)--cycle;

triple centroid = (0.4,0.6,0.466667);
draw(tri);
dot((0,0,0.7), blue);
label("$N$", (0,0,0.7), align=W);
dot((0.4,0.6,0.466667), blue);
draw(centroid--1.6centroid, arrow=Arrow3(DefaultHead2));
dot((0.45,1.5,1), blue);
label("$P$", (0.45,1.5,1), align=E);
draw((0.45,1.5,1)--(0.45,1.5,0), dotted);
