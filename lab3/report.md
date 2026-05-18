# Lab 3 Report: FEM Soft Body Simulation (Taichi)

The project is at https://github.com/dyc1111/PhySim2026Labs.

## 1. Project Structure

This project follows the same data/algorithm split as Lab 1 and Lab 2.

- `src/main.py` is the Hydra entry point. It initializes Taichi, reads the selected scene config, constructs `Scene`, builds the simulator, and starts the run loop.
- `src/scene.py` is the data side of the FEM solver. It owns vertex positions, velocities, forces, masses, pinned flags, element indices, rest-shape inverse matrices, deformation gradients, SVD factors, first Piola-Kirchhoff stress, and surface index buffers for rendering.
- `src/mesh.py` builds the two supported meshes. For the cuboid, it generates a tetrahedral grid and extracts outward-facing boundary triangles. For cloth, it generates a regular triangle grid embedded in 3D.
- `src/energy.py` contains the material model registry and the three hyperelastic models used in Bonus 1: StVK, Neo-Hookean, and Corotated. Each model writes the diagonal principal-stretch energy gradient into `Sgrad_3` or `Sgrad_2`.
- `src/simulator.py` is the algorithm side of the FEM solver. It owns the simulation loop, camera, rendering and pause/reset handling.
- `src/interaction.py` implements user interaction. With `CTRL + LMB`, the simulator selects the vertex nearest to the camera ray and applies a force proportional to mouse drag.
- `src/util.py` contains helper functions used by interaction.
- `cfg/base.yaml` selects the default scene. `cfg/scene/cuboid.yaml` and `cfg/scene/cloth.yaml` each contain both scene parameters and simulator parameters for the corresponding demo.

The simulation step is:

1. clear and set external forces
2. compute deformation gradients
3. compute SVD / principal stretches
4. apply the selected energy model
5. reconstruct first PK stress
6. scatter internal element forces to vertices
7. integrate velocity and position with explicit Euler

## 2. Basic Requirements: Cuboid Demo

The required demo is a 3D soft body simulation based on FEM.

### Simulation Pipeline

For each tetrahedron, `Scene.calc_grad()` computes:

$$
F = [x_1 - x_0,\ x_2 - x_0,\ x_3 - x_0]D_m^{-1}.
$$

The simulator computes the SVD:

$$
F = U\Sigma V^T,
$$

then the active energy model computes:

$$
\frac{\partial W}{\partial \sigma_i}.
$$

The first PK stress is reconstructed as:

$$
P = U\,\mathrm{diag}\left(\frac{\partial W}{\partial \sigma_i}\right)V^T.
$$

The element force matrix is:

$$
[f_1,\ f_2,\ f_3] = -V_e\,P\,D_m^{-T}, \qquad
f_0 = -f_1 - f_2 - f_3.
$$

Finally, the simulator uses explicit Euler:

$$
v \leftarrow v + hM^{-1}f,\qquad x \leftarrow x + hv.
$$

Pinned vertices are skipped during integration.

### Run command

Run the cuboid demo:

```bash
python src/main.py scene=cuboid
```

![Alt Text](gifs/cuboid.gif "cuboid")

## 3. Bonus 1: Energy Models

The implementation uses a material registry in `src/energy.py`:

```python
ENERGY_MODEL_REGISTER = {
    "StVK": StVk,
    "Neo-Hookean": NeoHookean,
    "Corotated": Corotated,
}
```

The simulator reads `model` from the config and constructs the corresponding material class. This keeps the rest of the FEM pipeline unchanged: the scene computes principal stretches, the material model writes `Sgrad`, and `Scene.calc_PK_stress()` reconstructs `P`.

All three models share the same interface:

```python
model.apply()
```

All three models write the diagonal matrix:

$$
\mathrm{diag}\left(\frac{\partial W}{\partial \sigma_i}\right).
$$

### StVK

StVK is stable for small and medium deformations, but is not robust under large rotations or inversion. In experiments, if a vertex is pushed into the body interior or the body flips inside-out, StVK may fail to generate a physically reasonable recovery force, so the object does not bounce back correctly.

### Neo-Hookean

Neo-Hookean fixes the inversion recovery issue, but is numerically more delicate because its formula contains terms like:

$$
\log J,\qquad \frac{1}{\sigma_i}.
$$

When a large external force collapses an element, one singular value can become very small. The stress then becomes extremely large, and explicit Euler can quickly diverge. In practice, a medium-to-large interaction force can make the Neo-Hookean system collapse.

### Corotated

Corotated elasticity was the most robust of the three in our experiments. Compared with StVK, it handles large rotations and inverted-looking configurations better. Compared with Neo-Hookean, it almost never diverge under strong interaction.

To switch models, use commands such as:

```bash
python src/main.py scene=cuboid scene.sim.model=Corotated
```

## 4. Bonus 2: Cloth Simulation

The cloth simulator reuses the same pipeline as the 3D soft body simulator:

1. build mesh
2. compute deformation gradient
3. compute principal stretches
4. evaluate the selected energy model
5. reconstruct first PK stress
6. scatter element forces
7. integrate vertices

The main difference is dimensionality. A tetrahedron has a 3D rest shape, so:

$$
F \in \mathbb{R}^{3 \times 3}.
$$

A cloth triangle is a 2D manifold embedded in 3D, so:

$$
F \in \mathbb{R}^{3 \times 2}.
$$

To keep one pipeline, each element uses the same 4-wide index field. Tetrahedra use all four entries, while cloth triangles use the first three and leave the fourth padded. The scene stores separate fields for the two deformation-gradient shapes:

```python
grad_3: 3 x 3
grad_2: 3 x 2
```

For cloth, the rest shape is represented by 2D coordinates. `src/mesh.py` precomputes:

$$
D_m^{-1} \in \mathbb{R}^{2 \times 2}, \qquad A = \frac{|\det(D_m)|}{2}.
$$

Then `Scene.calc_grad()` computes:

$$
F = [x_1 - x_0,\ x_2 - x_0]D_m^{-1}.
$$

Since Taichi's SVD works directly for square matrices but not for a `3 x 2` matrix, cloth computes the SVD through:

$$
C = F^TF \in \mathbb{R}^{2 \times 2}.
$$

The eigenvalues of `C` are squared singular values. The code takes their square roots to get the two principal stretches and reconstructs the thin left singular vectors:

$$
U = FV\Sigma^{-1}.
$$

This gives:

$$
P = U\,\mathrm{diag}\left(\frac{\partial W}{\partial \sigma_1}, \frac{\partial W}{\partial \sigma_2}\right)V^T
\in \mathbb{R}^{3 \times 2}.
$$

The triangle force matrix is:

$$
[f_1,\ f_2] = -A\,P\,D_m^{-T}, \qquad
f_0 = -f_1 - f_2.
$$

The cloth is rendered directly from its triangle faces. The config supports pinned boundaries and pinned corner patches; in practice, pinning small patches around corners is more stable than pinning only two single vertices, because point constraints create stress concentration in a pure membrane model.

Run the cloth demo:

```bash
python src/main.py scene=cloth
```

![Alt Text](gifs/cloth.gif "cloth")
