from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

cwd = Path.cwd()

divertor_data = cwd / "dane_od_Jorisa" / "divertor_mu2_mn11Stoerung.txt"
first_wall_data = cwd / "dane_od_Jorisa" / "LCFS+100-mu2-mn11Stoerung.txt"

firstwall = np.loadtxt(first_wall_data, delimiter=",")
divertor = np.loadtxt(divertor_data, delimiter=",")
# breakpoint()

x_divertor, y_divertor, z_divertor = divertor[:, 0], divertor[:, 1], divertor[:, 2]

fig = pv.Plotter()
fig.set_background("black")
fig.add_mesh(
    firstwall[:, :3],
    opacity=0.9,
    color="yellow",
)

fig.add_mesh(
    divertor[:, :3],
    color="red",
    render_points_as_spheres=True,
)


fig.show()
