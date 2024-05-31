import numpy as np
import pyvista as pv
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft, ifftshift


### powinno byc 51x51 endpoint false - should be then perfecly toroidally symmetric
step = 50
endpoint = False
Np = 5
plasma_surface = 98  # 0 - plasma axis, 98 - lcfs

assert 0 <= plasma_surface <= 98, "Incorrect plasma surface"


theta = np.linspace(0, 2 * np.pi, step, endpoint=endpoint)
phi = np.linspace(0, 2 * np.pi, step, endpoint=endpoint)
theta, phi = np.meshgrid(theta, phi)


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


Rmn_coeffs = np.array(Rmn_cos_coef).reshape(num_pol, num_tor, num_rad)[
    :, :, plasma_surface
]
Zmn_coeffs = np.array(Zmn_sin_coef).reshape(num_pol, num_tor, num_rad)[
    :, :, plasma_surface
]


def load_dist_field_data():
    cwd = Path.cwd()
    folder_name = f"theta-phi-{step}-{step}-{endpoint}-{endpoint}"
    full_path = cwd / "matlab_data" / "results" / folder_name
    f_name = (
        folder_name + "_BsBo.fld"
    )  ### bs/b0 - glowne wyniki - ostatnia kolumna to error field
    data = np.loadtxt(full_path / f_name, delimiter=",")
    return data


def calc_surface_fourier(
    phi,
    theta,
    Rmn_coeffs,
    Zmn_coeffs,
    Rmn_cos_num_pol,
    Rmn_cos_num_tor,
    save_file=False,
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

    def save_to_txt():
        df = pd.DataFrame(np.array((x.ravel(), y.ravel(), z.ravel())).T)
        columns = ["x", "y", "z"]
        df.columns = columns
        df.to_csv(
            f"theta-phi-{step}-{step}-{endpoint}-{endpoint}.txt", sep=",", index=None
        )

    if save_file:
        save_to_txt()
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
    normals_magnitude = np.sqrt(normals_x**2 + normals_y**2 + normals_z**2)

    normal_vectors = np.vstack(
        [normals_x.flatten(), normals_y.flatten(), normals_z.flatten()]
    ).T

    return normal_vectors, normals_magnitude


def plot_surface_and_normals(x, y, z, normal_vectors):
    fig = pv.Plotter()
    grid = pv.StructuredGrid(x, y, z)
    normal_vectors_normalized = normal_vectors / normals_magnitude.reshape(-1, 1)
    lcfs_points = np.array((x.flatten(), y.flatten(), z.flatten())).T
    fig.set_background("black")
    # fig.view_xy()
    fig.add_mesh(
        grid,
        opacity=0.9,
        color="green",
    )

    fig.add_mesh(
        lcfs_points,
        color="red",
        render_points_as_spheres=True,
    )
    # fig.add_arrows(
    #     lcfs_points, normal_vectors_normalized, mag=0.1, color="red", opacity=1
    # )
    fig.add_arrows(lcfs_points, normal_vectors, mag=0.1, color="orange", opacity=0.7)

    """Adds magnetic field vector"""
    B_err = add_magnetic_field_vector(fig)
    """Adds divertors"""
    # plot_divertor(fig)
    fig.show()
    return B_err


def add_magnetic_field_vector(fig):
    cwd = Path.cwd()
    full_path = cwd / "matlab_data" / "results" / "test"
    disturbed = np.loadtxt(
        full_path / f"theta-phi-{step}-{step}-{endpoint}-{endpoint}_disturbed.fld",
        delimiter=",",
    )
    undisturbed = np.loadtxt(
        full_path / f"theta-phi-{step}-{step}-{endpoint}-{endpoint}_initial.fld",
        delimiter=",",
    )
    undisturbed_vectors = undisturbed[:, -3:]
    disturbed_vectors = disturbed[:, -3:]

    def calculate_if_tangent_to_surface(normal_vectors, tangent_vectors, tolerance):
        ### sprawdzic, czy sa styczne poprzez pomnożenie normalnych do powierzchni
        ### do wektorow - jesli beda rowne 0 to znaczy ze sa styczne, powinny byc

        def dot_product(normal_vectors, tangent_vectors):
            dot_products = np.array(
                [
                    np.dot(normal_vectors[i], tangent_vectors[i])
                    for i in range(len(normal_vectors))
                ]
            )
            return dot_products

        # prod = dot_product(normal_vectors, tangent_vectors)
        # Checking if two vectors are perpendicular
        ####################################################################################################

        #### for 6e-2 wszystkie wartossci w zaleznosci od tego czy sa prostopadle wyrzucaja True

        #########################

        def are_perpendicular(normal_vectors, tangent_vectors, tol=tolerance):
            dot_products = dot_product(normal_vectors, tangent_vectors)
            # print(dot_products)
            return np.abs(dot_products) < tol

        are_perpendicular = are_perpendicular(normal_vectors, tangent_vectors)

        return are_perpendicular

    def vector_projection(v, n):
        """
        Oblicza projekcję wektora v na wektor n.

        :param v: Wektor do rzutowania (numpy array, rozmiar Nx3)
        :param n: Wektor normalny (numpy array, rozmiar Nx3)
        :return: Projekcja wektora v na wektor n (numpy array, rozmiar Nx3)
        """
        # Obliczanie iloczynu skalarnego dla każdego wektora
        v_dot_n = np.sum(v * n, axis=1)
        n_norm_sq = np.sum(n * n, axis=1)

        # Sprawdzamy, czy któryś wektor normalny nie jest wektorem zerowym
        if np.any(n_norm_sq == 0):
            raise ValueError("Wektor normalny nie może być wektorem zerowym.")

        # Obliczanie współczynników projekcji
        projection_factors = v_dot_n / n_norm_sq

        # Obliczanie projekcji
        projections = projection_factors[:, np.newaxis] * n

        return projections

    """
    ***************************************************************************
    ***************************************************************************
    ***************************************************************************    
    """

    tolerance = 6e-2
    undisturbed_tangent = calculate_if_tangent_to_surface(
        normal_vectors, undisturbed_vectors, tolerance
    )
    print(
        f"All vectors on a surface for undisturbed for tolerance {tolerance} -----",
        all(undisturbed_tangent),
    )

    disturbed_tangent = calculate_if_tangent_to_surface(
        normal_vectors, disturbed_vectors, tolerance
    )
    print(
        f"All vectors on a surface for disturbed for tolerance {tolerance} -----",
        all(disturbed_tangent),
    )

    ## Calculate projections on normal vectors
    dist_vect_projected_on_normal = vector_projection(disturbed_vectors, normal_vectors)

    ## Calculate magnitudes
    dist_projected_magnitude = np.linalg.norm(dist_vect_projected_on_normal, axis=1)
    undisturbed_magnitude = np.linalg.norm(undisturbed_vectors, axis=1)
    normal_vectors_magnitude = np.linalg.norm(normal_vectors, axis=1)

    ## Calculate Error Field
    B_err = dist_projected_magnitude / undisturbed_magnitude
    # B_err = dist_projected_magnitude / normal_vectors_mag1nitude

    breakpoint()

    """
    ***************************************************************************
    ***************************************************************************
    ***************************************************************************    
    """

    plot_magnetic_field_vectors(fig, undisturbed_vectors, disturbed_vectors)
    return B_err


def plot_magnetic_field_vectors(fig, undisturbed_vectors, disturbed_vectors):
    # fig = pv.Plotter()
    # grid = pv.StructuredGrid(x, y, z)
    lcfs_points = np.array((x.flatten(), y.flatten(), z.flatten())).T
    fig.set_background("black")
    fig.add_mesh(
        lcfs_points,
        color="red",
        opacity=0.1,
        render_points_as_spheres=True,
    )

    fig.add_arrows(lcfs_points, disturbed_vectors, mag=0.1, color="red", opacity=1)
    fig.add_arrows(
        lcfs_points, undisturbed_vectors, mag=0.2, color="white", opacity=0.3
    )


def plot_divertor(fig):
    cwd = Path.cwd()
    full_path = Path.cwd() / "matlab_data" / "results" / "test"

    f_name = "divertorx10clean.txt"  ### dane divertora
    data = np.loadtxt(full_path / f_name, delimiter=",")
    lcfs_points = data[:, :3]
    cloud = pv.PolyData(lcfs_points)
    cloud["point_color"] = data[:, -1]  # just use z coordinate
    fig.add_mesh(
        lcfs_points,
        color="black",
        # scalars="point_color",
        # cmap="viridis",
        render_points_as_spheres=True,
        point_size=5,
    )


def calc_fft(x, y, z, B_err, theta_dim, phi_dim):
    # data = load_dist_field_data()
    # dist_field = data[:, -1].reshape(-1, 1)
    # undist_field_magnitude = normals_magnitude.reshape(-1, 1)
    # field_err = (dist_field / undist_field_magnitude).reshape(theta_dim, phi_dim)
    field_err = B_err
    fft_result = np.fft.fft2(field_err)
    fft_result = np.fft.fftshift(fft_result)

    def plot_results():
        plt.figure(figsize=(13, 5))  # szerokość x wysokość
        plt.subplot(121)
        plt.imshow(field_err, cmap="viridis", aspect="auto")
        plt.colorbar(label="Aplitude")
        plt.title("Field Error")
        plt.xlabel("Theta")
        plt.ylabel("Phi")

        plt.subplot(122)
        plt.imshow(np.abs(fft_result), cmap="viridis", aspect="auto", norm="log")
        plt.colorbar(label="Amplitude")
        plt.title("FFT2")
        plt.xlabel("n (labels incorrect still)")
        plt.ylabel("m (labels incorrect still)")
        plt.show()

    plot_results()


if __name__ == "__main__":
    x, y, z = calc_surface_fourier(
        phi,
        theta,
        Rmn_coeffs,
        Zmn_coeffs,
        Rmn_cos_num_pol,
        Rmn_cos_num_tor,
        save_file=False,
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
    B_err = plot_surface_and_normals(x, y, z, normal_vectors)
    calc_fft(x, y, z, B_err.reshape((len(theta), len(phi))), len(theta), len(phi))
