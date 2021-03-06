// Autogenerated from examples/primal_introduction_ex

// To turn this Asymptote source file into an image for inclusion in
// Axom's documentation,
// 1. run Asymptote:
//    asy -f png showClip.asy
// 2. Optionally, use ImageMagick to convert the white background to transparent:
//    convert showClip.png -transparent white showClip.png

// preamble
settings.render = 6;
import three;
size(6cm, 0);

// axes
draw(O -- 1.7X, arrow=Arrow3(DefaultHead2), L=Label("$x$", position=EndPoint, align=W));
draw(O -- 2.4Y, arrow=Arrow3(), L=Label("$y$", position=EndPoint));
draw(O -- 2Z, arrow=Arrow3(), L=Label("$z$", position=EndPoint));

// polygon
path3 pgon = (0.533333,1,0)--(0,1,0.622222)--(0,0.514286,1)--(0.342857,0,1)--(1,0,0.233333)--(1,0.3,0)--cycle;

// triangle
path3 tri = (1.2,0,0)--(0,1.8,0)--(0,0,1.4)--cycle;

// draw triangle then polygon
draw(surface(tri), surfacepen=blue+opacity(0.4));
draw(tri);

draw(surface(pgon), surfacepen=yellow+opacity(0.4));
draw(pgon, yellow);

// bounding box
draw(box((0,-0.5,0), (1,1,1)));
