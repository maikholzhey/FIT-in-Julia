# ---- mh ----
# FIT in JULIA
# ------------
using LinearAlgebra
using Arpack
using SparseArrays
#using UnicodePlots
using Plots

# -------------------------------------------------------------------
# Use Atom Editor using JULIA plugin to keep session alive for speed
# -------------------------------------------------------------------

# time the script
t1 = time_ns()

# constants
c0 = 299792458;
mue0 =4e-7*pi;
eps0 = 1/(c0^2*mue0);

# geometry
nx = 161;
ny = nx;
nz = 2;
LxMax = (nx-1)*0.5; # 1/2 = 0 !integer division! WELCOME TO PYTHON
LyMax = (ny-1)*0.5;
LzMax = (nz-1)*0.5;
LxMin = -LxMax
LyMin = -LyMax
LzMin = -LzMax

# algebraic dimension
np = nx * ny * nz

# make mesh
xm = collect(range(LxMin, length = nx, stop = LxMax))
ym = collect(range(LyMin, length = ny, stop = LyMax))
zm = collect(range(LzMin, length = nz, stop = LzMax))

# createP
xp = ones(Int64,(np,))
Px = spdiagm(0 => -xp, 1 => xp[1:end-1])
Py = spdiagm(0 => -xp, nx => xp[1:end-nx])
Pz = spdiagm(0 => -xp, nx*ny => xp[1:end-nx*ny])

# create Operators
# ----------------
# Gradient G
G = [Px; Py; Pz]
# Divergence S
S = -G'
# curl C
zs = spdiagm(0 => 0*xp)
C = dropzeros([   zs     -Pz     Py;
				  Pz      zs    -Px;
				  -Py     Px     zs ])
# curl dual Cs
Cs = C'

# build Material
# --------------

# geometry
dx = [xm[2:end]-xm[1:end-1]; 0]
dy = [ym[2:end]-ym[1:end-1]; 0]
dz = [zm[2:end]-zm[1:end-1]; 0]

xd = (xm[1:end-1] + xm[2:end])/2 # length: nx-1
dxd = [xd[1]-xm[1]; xd[2:end]-xd[1:end-1]; xm[end]-xd[end]]# length nx

yd = (ym[1:end-1] + ym[2:end])/2 # length: ny-1
dyd = [yd[1]-xm[1]; yd[2:end]-yd[1:end-1]; ym[end]-yd[end]]# length ny

if nz != 1
	zd = (zm[1:end-1] + zm[2:end])/2 # length: ny-1
	if nz == 2
		dzd = [zd[1] - zm[1]; zm[end] - zd[end]]
	else
		dzd = [zd[1]-zm[1]; zd[2:end]-zd[1:end-1]; zm[end]-zd[end]]# length nz
	end
else
	# special for 2D problems: set dz = 1
	zd = zm
	dz = [1]
	dzd = [1]
end

### primary grid
dsx = repeat(dx,ny*nz,1)
dsy = reshape(repeat(dy',nx,nz),nx*ny*nz,1)
dsz = reshape(repeat(dz',nx*ny,1),nx*ny*nz,1)

ds = [dsx; dsy; dsz]
da = [dsy.*dsz ; dsx.*dsz ; dsx.*dsy]

### dual grid
dsdx = repeat(dxd,ny*nz,1)
dsdy = reshape(repeat(dyd',nx,nz),nx*ny*nz,1)
dsdz = reshape(repeat(dzd',nx*ny,1),nx*ny*nz,1)

dsd = [dsdx; dsdy; dsdz]
dad = [dsdy.*dsdz ; dsdx.*dsdz ; dsdx.*dsdy]


# nullinv to calculate material matrices
function nullinv(A)
	""" calculates pseudo inverse of input """
	nrow, ncol = size(A)
	B = zeros(nrow,ncol)
	tmpI = findall(x->x!=0,A)
	B[tmpI] = 1 ./ A[tmpI]
	return B
end

# boundary information
bc = [1 1 1 1 1 1] # 1 = PEC, 0 = PMC

epsilon = Array{Int64}(undef, 0)
mue = Array{Int64}(undef, 0)

if bc[1] == 1 # xlow
	iy=repeat(Array{Int64}(collect(range(1, length = ny, stop = ny))),nz,1)
    iz=reshape(ones(Int64,ny,1)*(Array{Int64}(collect(range(1, length = nz, stop = nz))))',ny*nz,1)
    ip = (iz.-1)*nx*ny .+ (iy.-1)*nx .+ 1
	epsilon = union(epsilon,[np.+ip;2*np.+ip]) # y,z-comp.
    mue = union(mue, ip)            # x  -comp.
end

if bc[2] == 1 # xhigh
	iy=repeat(Array{Int64}(collect(range(1, length = ny, stop = ny))),nz,1)
    iz=reshape(ones(Int64,ny,1)*(Array{Int64}(collect(range(1, length = nz, stop = nz))))',ny*nz,1)
    ip = (iz.-1)*nx*ny .+ (iy.-1)*nx .+ nx
	epsilon = union(epsilon,[np.+ip;2*np.+ip]) # y,z-comp.
    mue = union(mue, ip)            # x  -comp.
end

if bc[3] == 1 # ylow
	ix=repeat(Array{Int64}(collect(range(1, length = nx, stop = nx))),nz,1)
    iz=reshape(ones(Int64,nx,1)*(Array{Int64}(collect(range(1, length = nz, stop = nz))))',nx*nz,1)
    ip = (iz.-1)*nx*ny .+ (1-1)*nx .+ ix
	epsilon = union(epsilon,[ip;2*np.+ip]) # x,z-comp.
    mue = union(mue, np.+ip)            # y  -comp.
end

if bc[4] == 1 # yhigh
	ix=repeat(Array{Int64}(collect(range(1, length = nx, stop = nx))),nz,1)
    iz=reshape(ones(Int64,nx,1)*(Array{Int64}(collect(range(1, length = nz, stop = nz))))',nx*nz,1)
    ip = (iz.-1)*nx*ny .+ (ny-1)*nx .+ ix
	epsilon = union(epsilon,[ip;2*np.+ip]) # x,z-comp.
    mue = union(mue, np.+ip)            # y  -comp.
end

if bc[5] == 1 # zlow
	ix=repeat(Array{Int64}(collect(range(1, length = nx, stop = nx))),ny,1)
    iy=reshape(ones(Int64,nx,1)*(Array{Int64}(collect(range(1, length = ny, stop = ny))))',nx*ny,1)
    ip = (iy.-1)*nx .+ ix
	epsilon = union(epsilon,[ip;np.+ip]) # x,z-comp.
    mue = union(mue, 2*np.+ip)            # y  -comp.
end

if bc[6] == 1 # zhigh
	ix=repeat(Array{Int64}(collect(range(1, length = nx, stop = nx))),ny,1)
    iy=reshape(ones(Int64,nx,1)*(Array{Int64}(collect(range(1, length = ny, stop = ny))))',nx*ny,1)
    ip = (nz.-1)*nx*ny .+ (iy.-1)*nx .+ ix
	epsilon = union(epsilon,[ip;np.+ip]) # x,z-comp.
    mue = union(mue, 2*np.+ip)            # y  -comp.
end

# build Material matrices
# -----------------------
nds = nullinv(ds)
ndsd = nullinv(dsd)

# M eps
epsv0 = collect(range(eps0, length = 3*np, stop = eps0))
MeL = dad .* epsv0 .* nds
# boundary setting
MeL[epsilon] .= 0
MeLi = nullinv(MeL)
# build sparse diagonal matrices
Meps = spdiagm(0 => collect(Iterators.flatten(MeL)))
Mepsi = spdiagm(0 => collect(Iterators.flatten(MeLi)))

# M mue
muev0 = collect(range(mue0, length = 3*np, stop = mue0))
MmL = da .* muev0 .* ndsd
# boundary setting
MmL[mue] .= 0
MmLi = nullinv(MmL)
# build sparse diagonal matrices
Mmue = spdiagm(0 => collect(Iterators.flatten(MmL)))
Mmuei = spdiagm(0 => collect(Iterators.flatten(MmLi)))

# system matrix for stability
Acc = Mepsi * Cs * Mmuei * C
# calc eigenvalues and timestep
val, vect = eigs(Acc; nev=6, which=:LM, tol=0.0,)
dtmax = 2/sqrt(real(val[1]))
println(dtmax)

# canonical index of excitation

# Excitation
# ----------
function gaussin(tdt,f1,f2)
	"""calculates the gaussian excitation signal with a given spectrum"""
	lim1 = 1e-1  # spectrum factor at f1,f2
	lim2 = 1e-6  # error by signal step at it=1
	t0 = tdt[1]
	dt = tdt[2] - tdt[1]

	f0   = (f1+f2)/2
	df60 = abs(f2-f1)/2

	fg = df60 / sqrt(log(1/lim1))
	tpuls = sqrt(log(1/lim2)) /pi/fg

	# shift mid of signal to integer*dt
	tpuls = (floor(tpuls/ dt) + 1) * dt
	itpuls = Int(round(tpuls / dt)+1)

	# ======================================================================
	# symmetric modulated Gauss
	if abs(f0)<1e-12
		sig = MathConstants.e .^ ( -((tdt .- t0 .- tpuls) .* pi .* fg) .^ 2 )
	else
		sig = MathConstants.e .^ ( -((tdt .-t0 .- tpuls) .* pi .* fg) .^ 2 ) .* sin.(2 .* pi .* f0 .* (tdt .- t0 .- tpuls))
	end
	# ======================================================================

	# make symmetric signal end
	sig[(2*itpuls):end] .= 0
	npuls = 2 .* itpuls .- 1

	# norm to max = 1
	sig = sig ./ maximum(sig)

	return sig, npuls
end

Nt = 1400
ttt = collect(range(0, length = Nt, stop = Nt*dtmax))
f1 = 1e2
f2 = 6e6
# gaussin
sig, npuls = gaussin(ttt, f1, f2)

# work with ATOM editor and julia plugin!
#display(plot(ttt,sig,title="gaussin"))

# =========================
# MISC FUNCTIONS
# =========================
function dft()
	"""calculates the discrete fourier transform of a signal"""
end

function fieldProbe()
	"""gives a field value at a given edge in the mesh over time"""
end

function fieldMonitor()
	"""gives a field at a given frequency"""
end

function fieldEnergy()
	"""calculates the discrete electromagnetic energy"""
end
# =========================
# END OF MISC FUNCTIONS
# =========================

# canonical Index of exitation
zcI = Int(ceil(nx/2) + (ceil(ny/2)-1) * nx  + 2 * np)

# ----------
# Leapfrog
# ----------

# init
e = spzeros(3*np,1)
h = spzeros(3*np,1)
js = spzeros(3*np,1)
# discrete energy
em1 = e
hm1 = h
# dummy for field energy: iteration - 1
w = 0 .* ttt

# iterations
for i = 1:Nt
	global e, h , js, em1, hm1, dtmax, Mmue, Mmuei, Meps, Mepsi, w
	js[zcI] = sig[i]
	h = h .- dtmax * Mmuei * C * e
	e = e .+ dtmax * Mepsi * (Cs *h .- js)

	w[i] = (0.5 * (em1' * Meps * e + h' * Mmue * h))[1]
	# save n-1
	em1 = e
end

# calc overall time
t2 = time_ns()
total_time = (t2-t1)/1e9
println(total_time)

#println(e)
