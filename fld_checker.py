import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pathlib import Path
import json

cwd = Path.cwd()
full_path = cwd / "matlab_data" / "results" / "test"


def plot_field_vectors():
    disturbed = np.loadtxt(
        full_path / "theta-phi-50-50-False-False_disturbed.fld", delimiter=","
    )
    undisturbed = np.loadtxt(
        full_path / "theta-phi-50-50-False-False_initial.fld", delimiter=","
    )

    def calc_vectors_magnitude(vectors):
        vector_x = vectors[:, -3]
        vector_y = vectors[:, -2]
        vector_z = vectors[:, -1]

        vector_magnitude = np.sqrt(vector_x**2 + vector_y**2 + vector_z**2)
        return vector_magnitude

    disturbed_vector_magnitude = calc_vectors_magnitude(disturbed[:, -3:])

    undisturbed_vector_magnitude = calc_vectors_magnitude(undisturbed[:, -3:])

    from_matlab = disturbed
    x, y, z = from_matlab[:, 0], from_matlab[:, 1], from_matlab[:, 2]

    ### napisac funkcje czy punkty sa blisko

    fig = pv.Plotter()
    grid = pv.StructuredGrid(x, y, z)
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

    fig.add_arrows(lcfs_points, from_matlab[:, -3:], mag=0.2, color="red", opacity=1)
    fig.show()


def plot_field_errors():
    ### test for 10 divertor points
    # data = np.loadtxt(
    #     full_path / "test_symmetry" / "2x5_div" / "2x5_divertor_BsBo.fld",
    #     delimiter=",",
    # )

    #### high precision 100x100 - div 1.03 - finalne obliczenia
    # data = np.loadtxt(
    #     cwd
    #     / "matlab_data"
    #     / "1st_case_higher_precision"
    #     / "theta-phi-100-100-False-False_div_1.03_BsBo.fld",
    #     delimiter=",",
    # )

    data = np.loadtxt(
        cwd
        / "matlab_data"
        / "400x2000-caly_divertor"
        / "400x2000-caly_divertor_BsBo.fld",
        delimiter=",",
    )
    # data = np.loadtxt(
    #     cwd
    #     / "matlab_data"
    #     / "1st_case_higher_precision_2"
    #     / "50x300_False_div_1.03_BsBo.fld",
    #     delimiter=",",
    # )
    # #### first wall 100 outwards
    # data = np.loadtxt(
    #     cwd
    #     / "matlab_data"
    #     / "2nd_case_first_wall"
    #     / "First_wall_100mm_offset_mu1.03_4mmW_BsBo.fld",
    #     delimiter=",",
    # )

    ### test for 10 divertor points

    # ### calculated 50x50_False - with divertor
    # data = np.loadtxt(
    #     full_path / "theta-phi-50-50-False-False_BsBo.fld",
    #     delimiter=",",
    # )
    # ### calculated 50x50_False - with divertor

    def plot_pyvista():
        fig = pv.Plotter()
        fig.set_background("white")
        lcfs_points = data[:, :3]
        cloud = pv.PolyData(lcfs_points)
        cloud["point_color"] = data[:, -1]  # just use z coordinate
        fig.add_points(
            cloud,
            scalars="point_color",
            cmap="viridis",
            render_points_as_spheres=True,
            point_size=5,
        )
        fig.show()

    plot_pyvista()

    bsb0 = data[:, -1]
    field_err = bsb0.reshape((2000, 400)).T
    fft_result = np.fft.fft2(field_err, norm="ortho")
    # fft_result = np.fft.fftshift(fft_result)

    #### proba odwrocenia

    def plot_results():
        from matplotlib.ticker import ScalarFormatter

        plt.figure(figsize=(6, 5))  # szerokość x wysokość

        # plt.subplot(121)
        im1 = plt.imshow(field_err, cmap="viridis", aspect="auto")
        cbar1 = plt.colorbar(im1, label="Amplitude")
        cbar1.formatter = ScalarFormatter()
        cbar1.formatter.set_powerlimits((0, 0))
        cbar1.update_ticks()
        plt.title("Bs/Bo")
        plt.xlabel("Phi")
        plt.ylabel("Theta")

        # plt.subplot(122)
        # im2 = plt.imshow(np.abs(fft_result), cmap="viridis", aspect="auto")
        # cbar2 = plt.colorbar(im2, label="Amplitude")
        # cbar2.formatter = ScalarFormatter()
        # cbar2.formatter.set_powerlimits((0, 0))
        # cbar2.update_ticks()
        # plt.title("FFT2")
        # plt.xlabel("m")
        # plt.ylabel("n")

        plt.show()

    plot_results()


plot_field_errors()
