
#===============
 Pkg.add("PyPlot")
 Pkg.add("Interact")
    in Julia REPL
===============#
using Plots
pyplot()

# simple plot
plot(Plots.fakedata(50, 5), w=3)
# field on x,y mesh
x = 1:0.5:20
y = 1:0.5:10
f(x, y) = begin
(3x + y ^ 2) * abs(sin(x) + cos(y))
end
X = repeat(reshape(x, 1, :), length(y), 1)
Y = repeat(y, 1, length(x))
#Z = map(f, X, Y)
p1 = contour(x, y, f, fill=true)
#p2 = contour(x, y, Z)
plot(p1)#, p2)
