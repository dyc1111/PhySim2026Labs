# Lab 2 Report: Fluid Simulation (Taichi)

The project is at https://github.com/dyc1111/PhySim2026Labs.

## 1. Project Structure

This project is organized around a clear split between data and algorithms.

- `Scene` in `src/scene.py` is the data side. It owns all simulation state: staggered grid velocities (`grid_u`, `grid_v`, `grid_w`), previous grid velocities, grid density, cell types, solid velocities, particles, particle colors, APIC affine matrices, rigid-body states, and rendering buffers.
- `Simulator` in `src/simulator.py` is the algorithm side. It owns the simulation loop, GUI interaction, camera, rendering, video export, timestep control, and the step schedule.
- The concrete simulators `FpicSimulator`, `ApicSimulator`, and `EulerianFluidSimulator` all inherit from `Simulator` and only differ in how they build their strategy bundle.

The step dispatch is strategy-based. `src/strategy/__init__.py` defines:

```python
FluidStrategies(
    advection=...,
    collision=...,
    separation=...,
    transfer=...,
    density=...,
    divergence=...,
)
```

Then `Simulator._step(...)` calls the strategies in a fixed order:

1. rigid-body kinematics update
2. advection
3. collision
4. separation
5. collision again
6. particle/grid transfer (P2G)
7. density update
8. divergence-free projection
9. optional advection-reflection second half
10. grid/particle transfer (G2P)

Rigid bodies are integrated into the solver through `Scene`.

- `scene.update_cell_type()` marks boundary cells, water cells, and rigid-body occupied cells.
- It also writes `grid_solid_velocity`, so the fluid solver knows the solid velocity at solid cells/faces.
- `scene.pre_solve_kinematics(...)` and `scene.post_solve_kinematics(...)` update rigid linear and angular motion from user-applied forces and torques.
- Collision and pressure projection both read `grid_cell_type` and `grid_solid_velocity`, so rigid bodies affect both particle motion and grid projection.

## 2. FPIC Simulator

### Strategies

The FPIC simulator (`FpicSimulator`) uses:

- advection: `GravityIntegration`
- collision: `CollisionStrategy`
- separation: `SeparationStrategy` or `NoOpSeparationStrategy`
- transfer: `FpicTransferStragety`
- density: `DensityStrategy`
- divergence: `GaussSeidel`

### Interactive timestep

Interactive timestep editing is implemented in `src/simulator.py`.

- `Simulator._draw_gui()` creates a GUI sub-window with a `slider_float("dt", ...)`.
- The allowed range is `[1e-4, 5e-2]`.
- Each rendered frame reads the latest `dt`, then computes `sdt = dt / substeps`.
- Therefore the user can change `dt` while the simulator is running, and the new value is immediately used in subsequent substeps.

### PIC/FLIP interpolation

`FpicTransferStragety` supports interpolation between PIC and FLIP through the config field `flip_ratio` in `cfg/sim/fpic.yaml`.

During G2P:

- PIC interpolates the current grid velocity to the particle: `v_pic`
- FLIP interpolates the grid velocity change: `delta_v_flip = v_grid^{n+1} - v_grid^n`

The code updates particle velocity as:

$$
v_p \leftarrow \mathrm{flip\_ratio}\,\bigl(v_p + \Delta v_{\mathrm{flip}}\bigr) + \bigl(1 - \mathrm{flip\_ratio}\bigr)\,v_{\mathrm{pic}}
$$

So:

- `flip_ratio = 0` gives pure PIC
- `flip_ratio = 1` gives pure FLIP
- the default `flip_ratio = 0.95` gives the usual FLIP95 blend

### Gauss-Seidel in separation and divergence

The project uses iterative in-place relaxation in both the particle separation stage and the divergence-removal stage.

Particle separation in `src/strategy/separation.py` is an iterative local relaxation:

```text
for iter in range(num_particle_iters):
    rebuild spatial hash
    for each particle p:
        for each neighboring particle q:
            d = x_p - x_q
            dist = |d|
            if dist < 2 * radius:
                s = (radius - dist / 2) * d / (dist + eps)
                x_p += s
                x_q -= s
```

This is Gauss-Seidel-like because positions are updated immediately and later particle pairs see the newest values.

For incompressibility, `src/strategy/divergence.py` uses a red-black Gauss-Seidel update. The key idea is to split cells by parity:

```text
for iter in range(num_pressure_iters):
    for color in [0, 1]:
        for each water cell (x, y, z):
            if (x + y + z) % 2 != color:
                continue

            div = sum(outgoing face velocities with solid handling)
            n = number of non-solid neighbors
            if n == 0:
                continue

            div *= over_relaxation
            if compensate_drift and density > avg_density:
                div -= (density - avg_density)

            delta = div / n
            update the six neighboring staggered face velocities in place
```

The `mod2` split is important on GPU. Without it, adjacent cells could update the same face velocity at the same time and create write races. With red-black ordering, cells of one color are never face-adjacent, so the in-place Gauss-Seidel update becomes race-free inside one kernel launch.

### Run command

Run FPIC without rigid bodies:

```bash
python src/main.py sim=fpic scene=norigid
```

![Alt Text](gifs/fpic_norigid.gif "sim=fpic scene=norigid")

## 3. Eulerian Simulator

### Strategies

The Eulerian simulator (`EulerianFluidSimulator`) uses:

- advection: `SemiLagrangian`
- collision: `NoOpCollisionStrategy`
- separation: `NoOpSeparationStrategy`
- transfer: `EulerianTransferStrategy`
- density: `DensityStrategy`
- divergence: `LinearSystem`

### Semi-Lagrangian advection

The core Eulerian advection is in `src/strategy/advection.py`.

1. Read the current staggered grid velocities into `u_src`, `v_src`, `w_src`.
2. Build face metadata:
   - whether each face touches a solid cell
   - the solid velocity on that face
   - whether a face has known velocity because it is near water or solid
3. Extrapolate unknown face velocities outward for several iterations (`extrapolation_iters`) by averaging valid neighboring faces.
4. For every staggered face center `x0`, backtrace along the velocity field using RK2:

$$
\begin{aligned}
v_0 &= v(x_0) \\
x_{\mathrm{mid}} &= x_0 - \tfrac{1}{2}\,dt\,v_0 \\
v_{\mathrm{mid}} &= v(x_{\mathrm{mid}}) \\
x_{\mathrm{back}} &= x_0 - dt\,v_{\mathrm{mid}}
\end{aligned}
$$

5. Sample the old velocity field at `xback` using trilinear interpolation on the staggered grid.
6. Add gravity to the corresponding component.
7. Write the advected field back to `grid_u`, `grid_v`, `grid_w`.

### Advection-reflection

Advection-reflection is implemented in `src/simulator.py`.

When `sim.advection_reflection=True`:

- the code halves the substep: `sdt = dt / substeps / 2`
- it runs one half step of the normal pipeline
- it calls `scene.reflect()`, which performs

$$
u \leftarrow 2u - u_{\mathrm{prev}}, \qquad
v \leftarrow 2v - v_{\mathrm{prev}}, \qquad
w \leftarrow 2w - w_{\mathrm{prev}}.
$$

- it runs the same half-step pipeline a second time

So the simulator uses the first half-step result and the saved previous grid velocity to reflect the advection state, then re-advects once more.

### Pressure solve

The Eulerian divergence strategy in `src/strategy/divergence.py` builds and solves a sparse linear system for pressure.

1. Only water cells are assigned pressure unknowns.
2. For each water cell, the right-hand side is the cell divergence:

$$
b = \rho\,h\,\frac{\mathtt{div}}{dt}
$$

3. Neighbor handling:
   - water neighbor: add `-1` off-diagonal and `+1` to the diagonal
   - air neighbor: add `+1` only to the diagonal, which corresponds to free-surface pressure `p = 0`
   - solid neighbor: do not create a pressure unknown there; instead use solid face velocity in the divergence term and later enforce solid velocity during projection
4. Assemble `A` as a SciPy CSR sparse matrix.
5. Solve `Ap = b` with `AmgxSolver`, whose backend is `pyamgx`.
6. Project face velocities by subtracting `dt / rho * grad(p)` on fluid faces, while solid faces are overwritten by the rigid-body solid velocity.

This linear-system solve is the main bottleneck of the Eulerian method in this project. Compared with FPIC/APIC, it performs much heavier sparse assembly and external solver work, so it is much slower in practice.

### Run commands

Run Eulerian without rigid bodies, with advection-reflection enabled (default):

```bash
python src/main.py sim=eulerian scene=norigid
```

![Alt Text](gifs/eulerian_ar.gif "sim=eulerian scene=norigid sim.advection_reflection=True")

Run Eulerian without rigid bodies, with advection-reflection disabled:

```bash
python src/main.py sim=eulerian scene=norigid sim.advection_reflection=False
```

![Alt Text](gifs/eulerian.gif "sim=eulerian scene=norigid sim.advection_reflection=False")

## 4. APIC Simulator

APIC differs from FPIC only in the transfer strategy.

- FPIC uses `FpicTransferStragety`
- APIC uses `ApicTransferStrategy`

All other strategies are the same: gravity integration, collision, optional particle separation, density update, and Gauss-Seidel projection.

The APIC formulas shown in `image.png` are:

APIC, P2G:

$$
\begin{aligned}
(mv)_i^{n+1} &= \sum_p w_{i,p}\left[m_p v_p^n + m_p C_p^n \left(x_i - x_p^n\right)\right], \\
m_i^{n+1} &= \sum_p m_p w_{i,p}
\end{aligned}
$$

APIC, G2P:

$$
\begin{aligned}
v_p^{n+1} &= \sum_i w_{i,p} v_i^{n+1}, \\
C_p^{n+1} &= \frac{4}{(\Delta x)^2} \sum_i w_{i,p} v_i^{n+1} \left(x_i - x_p^n\right)^\top
\end{aligned}
$$

This matches the code in `src/strategy/transfer.py`:

- P2G transfers not only particle velocity `v_p`, but also the affine correction `C_p (x_i - x_p)`
- G2P reconstructs both the new particle velocity and the new affine matrix `particle_mat`

The weight function is the quadratic B-spline from `src/util/util.py`:

$$
B(x) =
\begin{cases}
0.75 - x^2, & |x| \le 0.5, \\
0.5\,(1.5 - |x|)^2, & 0.5 < |x| \le 1.5, \\
0, & \text{otherwise}.
\end{cases}
$$

Because APIC uses this wider B-spline support, its transfer loops iterate over a `4 x 4 x 4` neighborhood instead of the `2 x 2 x 2` linear interpolation stencil used by FPIC.

### Run command

Run APIC without rigid bodies:

```bash
python src/main.py sim=apic scene=norigid
```

![Alt Text](gifs/apic.gif "sim=apic scene=norigid")

## 5. Rigidbodies

Rigid-body support is implemented across `src/scene.py`, `src/rigidbody.py`, `src/interaction.py`, and the collision/divergence strategies.

### Interaction

The interaction method is:

- `Ctrl + left click`: translation
- `Ctrl + right click`: rotation

Implementation details:

- `InteractionHandler` casts a ray from the mouse into the scene.
- It picks the closest hit rigid body.
- Translation computes a force from mouse drag in camera right/up directions.
- Rotation computes a drag-induced torque from the clicked point and surface normal.
- The resulting force/torque arrays are passed into `scene.pre_solve_kinematics(...)`.

### Intersection detection

Intersection detection is based on solid occupancy tests over grid cells.

- For primitives (`Cuboid`, `Sphere`, `Cylinder`), the code uses analytic signed-distance-style tests in local coordinates.
- For `Custom`, the code loads a mesh, builds or loads an SDF cache, and samples the SDF with trilinear interpolation.
- `body.intersects_grid_cells(...)` returns whether each grid-cell center lies inside the rigid body up to a small inflate radius.
- `scene.update_cell_type()` then marks those cells as `CELL_SOLID`.

The custom-mesh path is the full SDF-based implementation. This is why arbitrary meshes such as `assets/bunny.stl` can be used as rigid bodies.

### Fluid behavior near rigid bodies

Fluid/rigid-body coupling follows the current cell classification.

- In particle collision handling, if a particle lies in a solid cell, its velocity is replaced by `grid_solid_velocity[cx, cy, cz]`.
- The code does not move that particle position out of the rigid body at this stage; only the velocity is changed.
- Therefore, for rigid-body collision, the fluid behavior is: same velocity as the rigid body, no position change.
- In the Eulerian pressure solve, solid-face velocities are also enforced during projection, so the grid solver is consistent with moving solids.

### Run commands

Run FPIC with the sphere scene:

```bash
python src/main.py sim=fpic scene=sphere
```

![Alt Text](gifs/fpic_sphere.gif "sim=fpic scene=sphere")

Run FPIC with the custom rigid-body scene:

```bash
python src/main.py sim=fpic scene=custom
```

![Alt Text](gifs/fpic_custom.gif "sim=fpic scene=custom")

## 6. Particle coloring

Particle coloring is density-based and implemented in `src/strategy/density.py`.

1. The code first scatters particle contributions to `grid_density` using trilinear weights.
2. On the first density pass, it computes `avg_density` as the average density over water cells.
3. Every frame, each particle color gradually fades toward blue:
   - red channel decreases slightly
   - green channel decreases slightly
   - blue channel increases slightly
4. Then the particle checks the density of its current grid cell.
5. If the relative density is low:

$$
\frac{\rho_{\mathrm{grid}}}{\rho_{\mathrm{avg}}} < 0.7
$$

the particle is recolored to a lighter blue:

$$
(0.6, 0.6, 1.0)
$$

So the policy highlights lower-density regions by making them brighter and lighter blue, while denser regions remain darker blue.
