import numpy as np
from pathlib import Path
import pyvista as pv
from scipy.spatial import cKDTree
from itertools import combinations
import math


def add_edge_columns(matrix):

    last_column = matrix[:, -1]  ### to jest tez endpoint
    new_matrix = np.column_stack((last_column, matrix))
    last_row = new_matrix[-1]  ### to jest tez endpoint
    matrix_extended = np.vstack((last_row, new_matrix))
    return matrix_extended


def connections_of_central_points(points, cols_nr_extended, rows_nr_extended):
    indices_of_points_to_connect = []
    indexes = []
    for i, cell in np.ndenumerate(points):
        if 0 in i or i[0] == rows_nr_extended - 1 or i[1] == cols_nr_extended - 1:
            continue
        p1_idx = (i[0] + 1, i[1] + 1)
        indexes.append(i)
        indices_of_points_to_connect.append([p1_idx])

    return indexes, indices_of_points_to_connect


def get_unique_indices_connections(
    indexes,
    indices_of_points_to_connect,
):
    indices_connections = []
    for index, cell in enumerate(indexes):
        for item in indices_of_points_to_connect[index]:
            indices_connections.append(sorted((cell, item)))
    unique_indices_sorted = {tuple(sublist) for sublist in sorted(indices_connections)}
    # return indices_connections
    return unique_indices_sorted


def get_mid_points(indices_connections, matrix_extended):
    mid_points = []
    for i in indices_connections:
        mid_points.append(float((matrix_extended[i[0]] + matrix_extended[i[1]]) / 2))
    return mid_points


def docelowe_dane():
    cols_nr_original = 121  ### w kierunku poloidalnym
    rows_nr_original = 721  ### w kierunku toroidalnym
    cols_nr_extended = cols_nr_original + 1
    rows_nr_extended = rows_nr_original + 1

    matrix_x = data[:, 0].reshape(rows_nr_original, cols_nr_original)
    matrix_y = data[:, 1].reshape(rows_nr_original, cols_nr_original)
    matrix_z = data[:, 2].reshape(rows_nr_original, cols_nr_original)

    matrix_extended_x = add_edge_columns(matrix_x)
    matrix_extended_y = add_edge_columns(matrix_y)
    matrix_extended_z = add_edge_columns(matrix_z)
    indexes_x, indices_of_points_to_connect_x = connections_of_central_points(
        matrix_extended_x, cols_nr_extended, rows_nr_extended
    )
    indexes_y, indices_of_points_to_connect_y = connections_of_central_points(
        matrix_extended_y, cols_nr_extended, rows_nr_extended
    )
    indexes_z, indices_of_points_to_connect_z = connections_of_central_points(
        matrix_extended_z, cols_nr_extended, rows_nr_extended
    )

    indices_connections_x = get_unique_indices_connections(
        indexes_x, indices_of_points_to_connect_x
    )
    indices_connections_y = get_unique_indices_connections(
        indexes_y, indices_of_points_to_connect_y
    )
    indices_connections_z = get_unique_indices_connections(
        indexes_z, indices_of_points_to_connect_z
    )
    mid_points_x = get_mid_points(indices_connections_x, matrix_extended_x)
    mid_points_y = get_mid_points(indices_connections_y, matrix_extended_y)
    mid_points_z = get_mid_points(indices_connections_z, matrix_extended_z)

    mid_points_dataset = np.array((mid_points_x, mid_points_y, mid_points_z)).T

    return mid_points_dataset


def find_nearest_neighbors(points1, points2, k=4):
    # Stwórz drzewo KD z drugiej tablicy punktów
    tree = cKDTree(points2)

    # Znajdź k najbliższych sąsiadów dla każdego punktu z pierwszej tablicy
    distances, indices = tree.query(points1, k=k)

    # Zwróć listę k najbliższych punktów dla każdego punktu z points1
    nearest_neighbors = points2[indices]
    return nearest_neighbors


def triangle_area_3d(A, B, C):
    AB = B - A
    AC = C - A
    cross_product = np.cross(AB, AC)
    area = 0.5 * np.linalg.norm(cross_product)

    ##################TODO SPRAWDZIC!!!!!!!!!!!!!!!!!!!!!!!!
    return area


def calculate_spheres():

    spheres = []
    for vertices in nearest_neighbors:
        # vertices_combinations = list(combinations(i, 2))
        # for idx in range(len(vertices) - 1):
        dist_1 = distance(vertices[0], vertices[1])
        dist_2 = distance(vertices[0], vertices[2])
        dist_3 = distance(vertices[0], vertices[3])

        distances = [dist_1, dist_2, dist_3]
        max_value = max(distances)
        max_index = distances.index(max_value)
        points_left_indices = [1, 2, 3]
        diagonal_point = points_left_indices.pop(max_index)
        # diagonal_line = np.vstack(
        #     (vertices[0].tolist(), vertices[diagonal_point].tolist())
        # )
        first_triangle_points = np.array(
            (vertices[0], vertices[points_left_indices[0]], vertices[diagonal_point])
        )
        second_triangle_points = np.array(
            (vertices[0], vertices[points_left_indices[1]], vertices[diagonal_point])
        )

        surface_area = triangle_area_3d(*first_triangle_points) + triangle_area_3d(
            *second_triangle_points
        )

        thickness = 0.004  # mm
        volume = float(surface_area * thickness)
        radius = (3 * volume / 4 / np.pi) ** (1 / 3)
        spheres.append(radius)
    return spheres


def test():
    cols_nr_original = 10
    rows_nr_original = 40

    cols_nr_extended = cols_nr_original + 1
    rows_nr_extended = rows_nr_original + 1

    matrix = np.arange(cols_nr_original * rows_nr_original).reshape(
        rows_nr_original, cols_nr_original
    )
    matrix_extended = add_edge_columns(matrix)

    indexes, indices_of_points_to_connect = connections_of_central_points(
        matrix_extended, cols_nr_extended, rows_nr_extended
    )
    indices_connections = get_unique_indices_connections(
        indexes, indices_of_points_to_connect
    )
    mid_points = get_mid_points(indices_connections, matrix_extended)
    # print(len(mid_points), len(indices_connections))

    return mid_points


def distance(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    d = math.sqrt(
        math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2) + math.pow(z2 - z1, 2) * 1.0
    )
    return d


def plot():

    fig = pv.Plotter()
    fig.set_background("black")
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    # fig.add_mesh(data, render_points_as_spheres=True, color="yellow")
    # fig.add_mesh(mid_points_dataset, color="red", render_points_as_spheres=True)
    # fig.add_mesh(
    #     nearest_neighbors[0][:2],
    #     color="purple",
    #     point_size=15,
    #     render_points_as_spheres=True,
    # )

    def plot_spheres():
        # fig.set_background("black")
        point_cloud = pv.PolyData(points)
        point_cloud["diameter"] = diameters
        spheres = point_cloud.glyph(scale="diameter", geom=pv.Sphere())
        fig.add_mesh(spheres, color="white", opacity=0.7)

    plot_spheres()
    fig.show()


if __name__ == "__main__":
    cwd = Path.cwd() / "calc_spheres/"
    data = np.loadtxt(cwd / "theta-phi-121-721_endpoint-True.txt", delimiter=",")

    mid_points_dataset = docelowe_dane()  ### 120x720 wierszy
    nearest_neighbors = find_nearest_neighbors(data, mid_points_dataset)
    spheres = calculate_spheres()
    data_with_spheres = np.concatenate((data, np.array(spheres).reshape(-1, 1)), axis=1)

    def reduce_matrix(data_with_spheres):
        data_with_spheres = data_with_spheres.reshape(721, 121, 4)
        data_with_spheres = data_with_spheres[:-1, :, :]
        data_with_spheres = data_with_spheres[:, :-1, :]
        return data_with_spheres

    data_with_spheres = reduce_matrix(data_with_spheres).reshape(-1, 4)

    def calculate_mu():
        phi_step = 720
        theta_step = 120
        phi_values = np.linspace(0, 2 * np.pi, phi_step, endpoint=False)
        theta_values = np.linspace(0, 2 * np.pi, theta_step, endpoint=False)
        # theta, phi = np.meshgrid(theta, phi)
        mu_list = []
        for phi in phi_values:
            for theta in theta_values:
                mu = 1.015 + 0.015 * np.cos(phi - theta)
                mu_list.append((mu))
        mu_list = np.array(mu_list)
        return mu_list

    mu_list = calculate_mu()
    data_with_spheres_and_mu = np.concatenate(
        (data_with_spheres, mu_list.reshape(-1, 1)), axis=1
    )

    points = data_with_spheres_and_mu[:, :3]
    diameters = data_with_spheres_and_mu[:, 3]
    from pathlib import Path

    cwd = Path.cwd()
    np.savetxt(
        cwd / "calc_spheres" / "spheres_and.txt",
        data_with_spheres_and_mu,
        delimiter=",",
    )
    plot()
