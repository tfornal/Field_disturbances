import numpy as np
import pyvista as pv
from pathlib import Path
import json

theta = np.linspace(0, 2 * np.pi, 50, endpoint=True)
phi = np.linspace(0, 2 * np.pi, 50, endpoint=True)
theta, phi = np.meshgrid(theta, phi)
Np = 5

dir_path = Path.cwd() / "fourier_coefficients" / "w7x_ref_1"

with open(dir_path / "rmn_data.json", "r") as file:
    Rmn_cos = json.load(file)
    Rmn_cos_num_pol = Rmn_cos.get("poloidal_mode_numbers")
    Rmn_cos_num_tor = Rmn_cos.get("toroidal_mode_numbers")
    Rmn_cos_num_rad = Rmn_cos.get("radial_points_number")
    Rmn_cos_coef = Rmn_cos.get("rmncos_coefficients")

with open(dir_path / "zmn_data.json", "r") as file:
    Zmn_sin = json.load(file)
    Zmn_sin_num_pol = Zmn_sin.get("poloidal_mode_numbers")
    Zmn_sin_num_tor = Zmn_sin.get("toroidal_mode_numbers")
    Zmn_sin_num_rad = Zmn_sin.get("radial_points_number")
    Zmn_sin_coef = Zmn_sin.get("rmncos_coefficients")

num_pol = len(Zmn_sin_num_pol)
num_tor = len(Zmn_sin_num_tor)
num_rad = Zmn_sin_num_rad


Rmn_coeffs = np.array(Rmn_cos_coef).reshape(num_pol, num_tor, num_rad)[:, :, -1]
Zmn_coeffs = np.array(Zmn_sin_coef).reshape(num_pol, num_tor, num_rad)[:, :, -1]


def load_dist_field_data():
    cwd = Path.cwd()
    full_path = cwd / "matlab_data" / "results" / "50x50"
    f_name = (
        "50x50_BsBo.fld"  ### bs/b0 - glowne wyniki - ostatnia kolumna to error field
    )
    data = np.loadtxt(full_path / f_name, delimiter=",")
    return data


def calc_surface_fourier(
    phi, theta, Rmn_coeffs, Zmn_coeffs, Rmn_cos_num_pol, Rmn_cos_num_tor
):
    R = 0
    Z = 0
    for pol_idx, m in enumerate(Rmn_cos_num_pol):
        for tor_idx, n in enumerate(Rmn_cos_num_tor):
            R += Rmn_coeffs[pol_idx][tor_idx] * np.cos(m * theta - n * Np * phi)
            Z += Zmn_coeffs[pol_idx][tor_idx] * np.sin(m * theta - n * Np * phi)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = Z
    return x, y, z


def diff_surface(phi, theta, Rmn_coeffs, Zmn_coeffs, Rmn_cos_num_pol, Rmn_cos_num_tor):
    dx_dphi = 0
    dy_dphi = 0
    dz_dphi = 0
    dx_dtheta = 0
    dy_dtheta = 0
    dz_dtheta = 0

    for pol_idx, m in enumerate(Rmn_cos_num_pol):
        for tor_idx, n in enumerate(Rmn_cos_num_tor):
            Rmn = Rmn_coeffs[pol_idx][tor_idx]
            Zmn = Zmn_coeffs[pol_idx][tor_idx]
            dx_dphi += -Np * Rmn * n * np.sin(Np * n * phi - m * theta) * np.cos(
                phi
            ) - Rmn * np.sin(phi) * np.cos(Np * n * phi - m * theta)
            dy_dphi += -Np * Rmn * n * np.sin(phi) * np.sin(
                Np * n * phi - m * theta
            ) + Rmn * np.cos(phi) * np.cos(Np * n * phi - m * theta)
            dz_dphi += -Np * Zmn * n * np.cos(Np * n * phi - m * theta)
            dx_dtheta += Rmn * m * np.sin(Np * n * phi - m * theta) * np.cos(phi)
            dy_dtheta += Rmn * m * np.sin(phi) * np.sin(Np * n * phi - m * theta)
            dz_dtheta += Zmn * m * np.cos(Np * n * phi - m * theta)
    return dx_dphi, dy_dphi, dz_dphi, dx_dtheta, dy_dtheta, dz_dtheta


def calc_normals(dx_dphi, dy_dphi, dz_dphi, dx_dtheta, dy_dtheta, dz_dtheta):
    normals_x = dy_dphi * dz_dtheta - dy_dtheta * dz_dphi
    normals_y = dx_dtheta * dz_dphi - dx_dphi * dz_dtheta
    normals_z = dx_dphi * dy_dtheta - dx_dtheta * dy_dphi
    normals_magnitude = np.sqrt(
        normals_x**2 + normals_y**2 + normals_z**2
    ).reshape(-1, 1)

    normal_vectors = np.vstack(
        [normals_x.flatten(), normals_y.flatten(), normals_z.flatten()]
    ).T

    return normal_vectors, normals_magnitude


def plot_surface_and_normals(normal_vectors):
    fig = pv.Plotter()
    grid = pv.StructuredGrid(x, y, z)
    normal_vectors_normalized = normal_vectors / normals_magnitude

    lcfs_points = np.array((x.flatten(), y.flatten(), z.flatten())).T
    fig.set_background("black")
    fig.add_mesh(
        grid,
        opacity=0.5,
        color="green",
    )
    fig.add_mesh(
        lcfs_points,
        color="red",
        render_points_as_spheres=True,
    )
    fig.add_arrows(
        lcfs_points, normal_vectors_normalized, mag=0.1, color="red", opacity=1
    )
    fig.add_arrows(lcfs_points, normal_vectors, mag=0.1, color="orange", opacity=0.7)
    fig.show()


def calc_disturbed_field_normal():
    # cwd = Path.cwd()
    # full_path = cwd / "matlab_data" / "results" / "50x50"
    # f_name = (
    #     "50x50_BsBo.fld"  ### bs/b0 - glowne wyniki - ostatnia kolumna to error field
    # )
    # data = np.loadtxt(full_path / f_name, delimiter=",")
    data = load_dist_field_data()
    dist_field = data[:, -1]
    dist_field_norm_comp = dist_field / np.linalg.norm(normal_vectors, axis=1)

    def plot_dist_field():
        fig = pv.Plotter()
        fig.set_background("grey")
        lcfs_points = data[:, :3]

        cloud = pv.PolyData(lcfs_points)
        cloud["Disturbed field"] = dist_field
        cloud["Disturbed field normal component"] = dist_field_norm_comp
        fig.add_points(
            cloud,
            scalars="Disturbed field normal component",
            cmap="viridis",
            render_points_as_spheres=True,
            point_size=8,
        )

        fig.add_points(
            cloud,
            scalars="Disturbed field",
            cmap="viridis",
            render_points_as_spheres=True,
            point_size=5,
        )
        fig.show()

    plot_dist_field()

    return dist_field, dist_field_norm_comp


def calc_fft(x, y, z, normals_magnitude):
    ### dist/undist
    data = load_dist_field_data()
    dist_field = data[:, -1]
    undist_field_magnitude = normals_magnitude
    field_err = dist_field.reshape(-1, 1) / undist_field_magnitude
    # breakpoint()
    # ### przeprowadzic fft2 dla field_err <- zrozumiec

    # grid_x = x
    # grid_y = y
    # grid_z = z
    # pass


if __name__ == "__main__":
    x, y, z = calc_surface_fourier(
        phi,
        theta,
        Rmn_coeffs,
        Zmn_coeffs,
        Rmn_cos_num_pol,
        Rmn_cos_num_tor,
    )
    dx_dphi, dy_dphi, dz_dphi, dx_dtheta, dy_dtheta, dz_dtheta = diff_surface(
        phi,
        theta,
        Rmn_coeffs,
        Zmn_coeffs,
        Rmn_cos_num_pol,
        Rmn_cos_num_tor,
    )
    normal_vectors, normals_magnitude = calc_normals(
        dx_dphi,
        dy_dphi,
        dz_dphi,
        dx_dtheta,
        dy_dtheta,
        dz_dtheta,
    )

    calc_fft(x, y, z, normals_magnitude)
    plot_surface_and_normals(normal_vectors)
    calc_disturbed_field_normal()
