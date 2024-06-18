import numpy as np
import pyvista as pv
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft, ifftshift
from matplotlib.ticker import ScalarFormatter


### powinno byc 51x51 endpoint false - should be then perfecly toroidally symmetric
step = 100
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


def calc_surf_coordinates(
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


def calc_offset_surface(
    x, y, z, normal_vectors, normals_magnitude, offset_distance=100 / 1000
):
    normals_normalized = normal_vectors / normals_magnitude.flatten()[:, np.newaxis]
    x_offset = x.flatten() + normals_normalized[:, 0] * offset_distance
    y_offset = y.flatten() + normals_normalized[:, 1] * offset_distance
    z_offset = z.flatten() + normals_normalized[:, 2] * offset_distance
    return x_offset, y_offset, z_offset


def is_tangent_to_surface(normal_vectors, tangent_vectors):
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

    dot = dot_product(normal_vectors, tangent_vectors)

    def are_perpendicular(normal_vectors, tangent_vectors, tol=1e-6):
        dot = dot_product(normal_vectors, tangent_vectors)
        # print(dot_products)
        return np.abs(dot) < tol

    are_perpendicular = are_perpendicular(normal_vectors, tangent_vectors)

    return dot, are_perpendicular


def vector_projection_to_calc_vectors(v, n):
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
    projected_vectors_in_normal_dir = projection_factors[:, np.newaxis] * n

    return projected_vectors_in_normal_dir


def vector_projection_new_function(v, n):
    dot_product_vn = np.dot(v, n)
    dot_product_nn = np.dot(n, n)
    projection = dot_product_vn / np.sqrt(dot_product_nn)  # * n

    ### second approach
    # dot_product_nn2 = np.linalg.norm(n)
    # projection2 = dot_product_vn / dot_product_nn2  # * n
    return projection


def read_vector_data_from_matlab():
    cwd = Path.cwd()

    # full_path = cwd / "matlab_data" / "results" / "test" / "test_symmetry"
    full_path = cwd / "matlab_data" / "1st_case_higher_precision"

    def divertor_50_50_input_data():
        disturbed_vectors = np.loadtxt(
            full_path / f"theta-phi-{step}-{step}-{endpoint}-{endpoint}_disturbed.txt",
            delimiter="\t",
        )
        undisturbed_vectors = np.loadtxt(
            full_path / f"theta-phi-{step}-{step}-{endpoint}-{endpoint}_initial.txt",
            delimiter="\t",
        )
        return disturbed_vectors, undisturbed_vectors

    def test_points_10_symmetrical():
        disturbed_vectors = np.loadtxt(
            cwd
            / "matlab_data"
            / "results"
            / "test"
            / "test_symmetry"
            / "2x5_div_high_prec"
            / "2x5_divertor_disturbed.fld",
            delimiter=",",
        )
        undisturbed_vectors = np.loadtxt(
            cwd
            / "matlab_data"
            / "results"
            / "test"
            / "test_symmetry"
            / "2x5_div_high_prec"
            / "2x5_divertor_initial.fld",
            delimiter=",",
        )
        return disturbed_vectors, undisturbed_vectors

    def first_case():
        full_path = cwd / "matlab_data" / "1st_case_higher_precision"
        disturbed_vectors = np.loadtxt(
            full_path / f"theta-phi-100-100-False-False_div_1.03_disturbed.fld",
            delimiter=",",
        )
        undisturbed_vectors = np.loadtxt(
            full_path / f"theta-phi-100-100-False-False_div_1.03_initial.fld",
            delimiter=",",
        )
        return disturbed_vectors, undisturbed_vectors

    def second_case_first_wall():
        full_path = cwd / "matlab_data" / "2nd_case_first_wall"
        disturbed_vectors = np.loadtxt(
            full_path / f"First_wall_100mm_offset_mu1.03_4mmW_disturbed.fld",
            delimiter=",",
        )
        undisturbed_vectors = np.loadtxt(
            full_path / f"First_wall_100mm_offset_mu1.03_4mmW_initial.fld",
            delimiter=",",
        )
        return disturbed_vectors, undisturbed_vectors

    # disturbed_vectors, undisturbed_vectors = divertor_50_50_input_data()
    # disturbed_vectors, undisturbed_vectors = test_points_10_symmetrical()
    disturbed_vectors, undisturbed_vectors = first_case()
    # disturbed_vectors, undisturbed_vectors = second_case_first_wall()

    return undisturbed_vectors, disturbed_vectors


def calc_error_field(undisturbed_data, disturbed_data):
    #### get magnetic field vectors
    undisturbed_vectors = undisturbed_data[:, -3:]
    disturbed_vectors = disturbed_data[:, -3:]
    # substracted = undisturbed_vectors - disturbed_vectors

    substracted = disturbed_vectors - undisturbed_vectors
    substracted_projected_vectors_in_normal_direction = (
        vector_projection_to_calc_vectors(substracted, normal_vectors)
    )

    def use_new_function():
        substracted_projected = []
        for i, normal_vector in enumerate(normal_vectors):
            projected = vector_projection_new_function(substracted[i], normal_vector)
            substracted_projected.append(projected)
        substracted_projected = np.array(substracted_projected)
        return substracted_projected

    substracted_projected = use_new_function()
    undisturbed_magnitude = np.linalg.norm(undisturbed_vectors, axis=1)

    ## Calculate Error Field
    B_err = substracted_projected / undisturbed_magnitude
    B_err = B_err.reshape((len(theta), len(phi))).T

    import pickle

    # Create a 100x100 matrix (example: random values)
    # matrix = np.random.rand(100, 100)

    # Define the file path for the pickle file
    file_path = "B_err.pkl"

    # Save the matrix to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(B_err, f)

    print(f"Matrix has been saved to {file_path}")

    breakpoint()

    return substracted_projected_vectors_in_normal_direction, B_err


def plot_magnetic_field_vectors(fig, undisturbed_vectors, disturbed_vectors):
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
    return fig


def plot_projected_vectors(fig, projected_vectors):
    magnitude_projected_vectors = 2000
    lcfs_points = np.array((x.flatten(), y.flatten(), z.flatten())).T
    fig.add_arrows(
        lcfs_points,
        projected_vectors,
        mag=magnitude_projected_vectors,
        color="red",
        opacity=1,
    )
    return fig


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


def od_jorisa(fig):

    cwd = Path.cwd()

    divertor_data = cwd / "dane_od_Jorisa" / "divertor_mu2_mn11Stoerung.txt"
    first_wall_data = cwd / "dane_od_Jorisa" / "LCFS+100-mu2-mn11Stoerung.txt"

    firstwall = np.loadtxt(first_wall_data, delimiter=",")
    divertor = np.loadtxt(divertor_data, delimiter=",")

    ### firstwall
    fig.add_mesh(
        firstwall[:, :3],
        color="orange",
        render_points_as_spheres=True,
    )

    ### divertor with fixed sphere sizes
    point_cloud = pv.PolyData(divertor[:, :3])
    diameters = divertor[:, -2]
    point_cloud["diameter"] = diameters

    spheres = point_cloud.glyph(scale="diameter", geom=pv.Sphere())
    fig.add_mesh(spheres, color="white", opacity=0.7)


def plot_offset_points(fig):
    x_offset, y_offset, z_offset = calc_offset_surface(
        x, y, z, normal_vectors, normals_magnitude
    )
    offset_points = np.array((x_offset, y_offset, z_offset)).T
    fig.add_mesh(
        offset_points,
        color="red",
        render_points_as_spheres=True,
    )


def plot_w7x(x, y, z, normal_vectors, plot=False):
    fig = pv.Plotter()
    grid = pv.StructuredGrid(x, y, z)

    lcfs_points = np.array((x.flatten(), y.flatten(), z.flatten())).T
    fig.set_background("black")
    fig.view_xy()
    fig.add_mesh(
        grid,
        opacity=0.4,
        color="green",
    )

    fig.add_mesh(
        lcfs_points,
        color="yellow",
        render_points_as_spheres=True,
    )

    """Dodaje info z plikow od jorisa"""
    od_jorisa(fig)

    """Adds offset points"""
    # plot_offset_points(fig)

    """Adds normal vectors"""
    fig.add_arrows(lcfs_points, normal_vectors, mag=0.1, color="orange", opacity=0.1)

    """Adds projected vectors"""
    plot_projected_vectors(fig, projected_vectors_in_normal_dir)
    #
    """Adds magnetic field vector"""
    # plot_magnetic_field_vectors(fig, undisturbed_data[:, -3:], disturbed_data[:, -3:])

    """Adds divertors"""
    # plot_divertor(fig)

    if plot:
        fig.show()


def calc_fft(B_err):  ##### nowa wersja -
    fft_result = np.fft.fft2(B_err)

    n, m = B_err.shape
    nr_of_points = n * m
    normalized_fft_data = fft_result / (np.sqrt(nr_of_points))
    # fft_result_shifted = np.fft.fftshift(normalized_fft_data)

    # Define the file path for the pickle file
    file_path = "FFT2_of_Berr.pkl"
    import pickle

    # Save the matrix to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(normalized_fft_data, f)

    print(f"Matrix has been saved to {file_path}")

    fft_result_inversed = np.fft.ifft2(fft_result)

    return normalized_fft_data, fft_result_inversed
    # return fft_result_shifted, fft_result_inversed


def plot_fft(field_err, fft_result_shifted, fft_result_inversed):

    plt.figure(figsize=(13, 5))  # szerokość x wysokość
    # plt.suptitle("Normal components - First_wall = 100mm; mu = 1.03, W = 4mm")
    plt.suptitle("Normal components - 100x100; False endpoint")
    plt.subplot(121)
    im1 = plt.imshow(field_err, cmap="viridis", aspect="auto")
    cbar1 = plt.colorbar(im1, label="Amplitude")
    cbar1.formatter = ScalarFormatter()
    cbar1.formatter.set_powerlimits((0, 0))
    cbar1.update_ticks()
    plt.title("B_err")
    plt.xlabel("Phi")
    plt.ylabel("Theta")

    plt.subplot(122)
    im2 = plt.imshow(
        np.abs(fft_result_shifted), cmap="viridis", aspect="auto", norm="log"
    )
    cbar2 = plt.colorbar(im2, label="Amplitude")
    cbar2.formatter = ScalarFormatter()
    cbar2.formatter.set_powerlimits((0, 0))
    cbar2.update_ticks()
    plt.title("FFT2")
    plt.xlabel("m")
    plt.ylabel("n")
    plt.show()


if __name__ == "__main__":
    x, y, z = calc_surf_coordinates(
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

    ### calculate offset
    # x_offset, y_offset, z_offset = calc_offset_surface(
    #     x, y, z, normal_vectors, normals_magnitude
    # )

    ### get matlab data
    undisturbed_data, disturbed_data = read_vector_data_from_matlab()

    ### substract vectors
    projected_vectors_in_normal_dir, B_err = calc_error_field(
        undisturbed_data, disturbed_data
    )
    plot_w7x(x, y, z, normal_vectors, plot=True)

    ### FFT2
    fft_result_shifted, fft_result_inversed = calc_fft(B_err)
    plot_fft(B_err, fft_result_shifted, fft_result_inversed)
