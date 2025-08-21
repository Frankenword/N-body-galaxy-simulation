import numpy as np
import os
import matplotlib.pyplot as plt
from frontend import *


# Set values for constants and required variables

N_particles = 500
L = 2.0                    # box side length
G = 1.0                    # corrected gravitational constant
Mtot = 1.0
mass = np.ones(N_particles) * (Mtot / N_particles)

dt = 0.01                  # timestep
T_end = 3.0                # total time
n_steps = int(T_end / dt)

dt_out = 0.05              # output interval
out_every = round(dt_out / dt)

# 30 equally spaced bins between 0 and 1
n_bins = 30
r_max_xi = 1.0
bin_edges = np.linspace(0.0, r_max_xi, n_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Softening
eps = 0.05

# Output folders
snap_dir = "snapshots"
xi_dir = "xi_output"
os.makedirs(snap_dir, exist_ok=True)
os.makedirs(xi_dir, exist_ok=True)


# generate arrays for particle positions and velocities

r = np.random.uniform(-L/2, +L/2, size=(N_particles, 3))
v = np.random.uniform(-0.1, +0.1, size=(N_particles, 3))


# define functions for: reentry, 26 copies, energy, momentum, xi(r) = autocorrelation function

def reentry(r, L):
    return (r + L/2) % L - L/2

def make_27_copies(positions, L):
    shifts = np.array([(i, j, k) for i in (-L, 0.0, L)
                                for j in (-L, 0.0, L)
                                for k in (-L, 0.0, L)], dtype=float)
    src = shifts[:, None, :] + positions[None, :, :]
    return src.reshape(-1, 3)

def minimum_image(d, L):
    return d - L * np.round(d / L)

def total_energy(r, v, m, L, G, eps):
    KE = 0.5 * np.sum(m * np.sum(v*v, axis=1))
    PE = 0.0
    N = len(r)
    for i in range(N):
        for j in range(i+1, N):
            d = minimum_image(r[i] - r[j], L)
            dist = np.sqrt(np.dot(d, d) + eps*eps)
            PE -= G * m[i] * m[j] / dist
    return KE + PE

def total_momentum(v, m):
    return np.sum(m[:, None] * v, axis=0)

def xi_of_r(r, L, bin_edges):
    N = len(r)
    dists = []
    for i in range(N):
        di = r[i] - r[i+1:]
        di = minimum_image(di, L)
        ds = np.linalg.norm(di, axis=1)
        dists.append(ds)
    dists = np.concatenate(dists) if dists else np.array([])

    dists = dists[(dists >= bin_edges[0]) & (dists < bin_edges[-1])]
    counts, _ = np.histogram(dists, bins=bin_edges)

    V = L**3
    shell_vol = (4.0/3.0)*np.pi*(bin_edges[1:]**3 - bin_edges[:-1]**3)
    total_pairs = N*(N-1)/2
    expected = total_pairs * (shell_vol / V)

    with np.errstate(divide='ignore', invalid='ignore'):
        xi = counts / expected - 1.0
        xi[np.isnan(xi)] = 0.0
    return xi

def save_snapshot(k, r):
    np.savetxt(f"{snap_dir}/snapshot{k:04d}.dat", r, fmt="%.6f")

def save_xi(k, r_centers, xi):
    np.savetxt(f"{xi_dir}/xi_{k:03d}.dat",
               np.column_stack([r_centers, xi]),
               fmt="%.6f",
               header="r  xi(r)")


# Leapfrog integration

r += 0.5 * dt * v
r = reentry(r, L)
softening_target = np.full(N_particles, eps)

for step in range(n_steps):
    sources_pos = make_27_copies(r, L)
    sources_mass = np.tile(mass, 27)
    softening_source = np.full(len(sources_pos), eps)

    a = AccelTarget(
        rmax=L,
        pos_target=r,
        pos_source=sources_pos,
        m_source=sources_mass,
        softening_target=softening_target,
        softening_source=softening_source,
        G=G,
        theta=0.5,
        parallel=False,
        tree=None,
        return_tree=False,
        method="adaptive",
        quadrupole=False
    )

    v += dt * a
    r += dt * v
    r = reentry(r, L)

    # Output at every dt_out
    if step % out_every == 0:
        out_idx = step // out_every + 1
        tnow = step * dt

        save_snapshot(out_idx, r)

        xi_vals = xi_of_r(r, L, bin_edges)
        save_xi(out_idx, bin_centers, xi_vals)

        E = total_energy(r, v, mass, L, G, eps)
        P = total_momentum(v, mass)
        print(f"t={tnow:.2f}  E={E:.6f}  P=({P[0]:.6f},{P[1]:.6f},{P[2]:.6f})")


# plot autocorrelation function at T = 0, 1, 2, 3

time_points = [0.0, 1.0, 2.0, 3.0]
out_indices = [int(t / dt_out) for t in time_points]
out_indices[-1] = min(out_indices[-1], n_steps // out_every)  # cap at last saved file

plt.figure()
for t,k in zip(time_points, out_indices):
    file_idx = k if k > 0 else 1  # ensure first file is xi_001.dat
    if file_idx > 60:  # check
        file_idx = 60
    data = np.loadtxt(f"{xi_dir}/xi_{file_idx:03d}.dat")
    rvals, xivals = data[:, 0], data[:, 1]
    plt.plot(rvals, xivals, label=f"T = {t:.0f}")
plt.xlabel("r")
plt.ylabel("Xi(r)")
plt.title("Autocorrelation Xi(r) at T = 0, 1, 2, 3")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("xi_times_0_1_2_3.png", dpi=200)
