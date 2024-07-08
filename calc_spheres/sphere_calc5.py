import numpy as np
from pathlib import Path
import pyvista as pv


##### DZIALAJACA TESTOWA WERSJA
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


##### DZIALAJACA TESTOWA WERSJA
cwd = Path.cwd() / "calc_spheres/"
data = np.loadtxt(cwd / "theta-phi-121-721_endpoint-True.txt", delimiter=",")


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

    ##### TRZEBA JESZCZE USUNAC DUPLIKATY
    def plot():

        fig = pv.Plotter()
        fig.set_background("black")
        x, y, z = data[:, 0], data[:, 1], data[:, 2]

        # nx, ny = 121, 721  # Wymiary siatki
        # x = x.reshape((nx, ny))
        # y = y.reshape((nx, ny))
        # z = z.reshape((nx, ny))

        # grid = pv.StructuredGrid(x, y, z)
        # fig.add_mesh(
        #     grid,
        #     opacity=0.1,
        #     color="green",
        # )
        fig.add_mesh(data, render_points_as_spheres=True)
        print(len(mid_points_dataset), data.shape)
        fig.add_mesh(mid_points_dataset, color="red", render_points_as_spheres=True)
        fig.show()

    plot()
    return mid_points_dataset


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
    print(len(mid_points), len(indices_connections))
    return mid_points


if __name__ == "__main__":
    # mid_points = test()
    mid_points_dataset = docelowe_dane()
