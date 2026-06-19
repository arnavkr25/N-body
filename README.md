# N-body
# N-Body Gravitational Simulator

A 2D gravitational N-body simulator built iteratively in Python, evolving from a basic Euler integrator to a fully optimized Barnes-Hut + RK4 simulation. Each version adds a meaningful physical or computational improvement over the previous one.

---

## Versions

### `nbody1.py` — Euler Integrator (Baseline)
The starting point. Computes gravitational forces between all pairs of bodies using a direct O(N²) summation and integrates motion with a simple Euler method. A small softening parameter `ε = 1e-5` is added to the distance to avoid singularities at close encounters.

**Limitation:** Euler integration is first-order accurate. Energy is not conserved — orbits visibly spiral outward over time.

---

### `nbody2.py` — Runge-Kutta 4th Order Integration
Replaces the Euler integrator with RK4. At each timestep, four force evaluations (k1–k4) are made at intermediate positions and velocities, and the updates are combined with the standard (1, 2, 2, 1)/6 weighting.

**Why RK4?** Euler was visibly leaking energy — orbits that should have been closed were slowly spiraling out. RK4's fourth-order accuracy (local error ~O(dt⁵) vs Euler's ~O(dt²)) fixes this without needing a much smaller timestep.

---

### `nbody3.py` — RK4 + Energy Tracking
Adds a live energy conservation plot alongside the orbital view. At each frame, total mechanical energy (KE + PE) is computed and plotted against simulation time.

**Why track energy?** Total energy is a conserved quantity in an isolated gravitational system. Monitoring its drift is the standard diagnostic for integration quality — a flat energy plot means the integrator is doing its job.

---

### `nbody4.py` — Barnes-Hut Tree + RK4 (51 bodies)
Introduces the Barnes-Hut algorithm to reduce force computation from O(N²) to O(N log N). A quadtree is built every timestep; for each body, distant clusters are approximated by their center of mass using the opening angle criterion `s/d < θ` (θ = 0.5). Scales to 50 orbiting bodies around a central mass.

**How the tree works:**
1. All bodies are inserted into a quadtree whose root spans the full particle distribution.
2. For each body i, the tree is traversed recursively. If a node is far enough away (size/distance < θ), its total mass and center of mass are used as a single approximation instead of summing over every body inside it.
3. The Barnes-Hut criterion trades a small, controllable force error for a large speedup — exact at θ = 0, increasingly approximate (and fast) as θ grows.

**Softening parameter:** Raised to `ε = 0.1` to handle the denser particle distribution without singularities on close passes.

---


## Physics Summary

The gravitational acceleration on body i due to all others is:

$$\vec{a}_i = \sum_{j \neq i} \frac{G m_j (\vec{r}_j - \vec{r}_i)}{|\vec{r}_j - \vec{r}_i|^3 + \varepsilon}$$

Initial conditions for the disk simulations (v5, v6) place each orbiting body at a random radius `r ∈ [5, 50]` with a circular orbit velocity `v = √(GM/r)`, giving a stable rotating disk around the central mass.

---

## Dependencies

```
numpy
matplotlib
```

Install with:
```
pip install numpy matplotlib
```

---

## Running

```bash
python nbody1.py   # Euler baseline
python nbody3.py   # RK4
python nbody4.py   # RK4 + energy plot
python nbody5.py   # Barnes-Hut, 51 bodies
```

Each script opens a live matplotlib animation window. v4 and onwards show an energy conservation plot on the right panel.

---

## Results

Energy drift (nbody3): < 0.01 units over 1000 frames at dt = 0.01, confirming RK4 integration quality.

nbody5 sustains 51-body dynamics with a Barnes-Hut quadtree (θ = 0.5) rebuilt at each RK4 stage.
