import numpy as np
import pyvista as pv
from pathlib import Path

# Parametry walca
radius = 1.0
height = 5.0
num_circumferential_points = 120  # liczba punktów wzdłuż obwodu
num_height_points = 720  # liczba punktów wzdłuż wysokości

# Obliczenie powierzchni bocznej walca
cylinder_surface_area = 2 * np.pi * radius * height

# Obliczenie powierzchni jednego kafelka
tile_width = (2 * np.pi * radius) / num_circumferential_points
tile_height = height / num_height_points
tile_area = tile_width * tile_height

# Liczba punktów w siatce
num_points = num_circumferential_points * num_height_points

cwd = Path.cwd()

# Wczytywanie punktów z pliku
try:
    points = np.loadtxt(cwd / "calc_spheres" / "720x120-offset_50.txt", delimiter=",")[
        :num_points
    ]
except Exception as e:
    print(f"Error loading points from file: {e}")
    exit(1)

# Sprawdzenie, czy dane są w poprawnym formacie
if points.shape[1] != 3:
    print("Error: points file must have exactly three columns (x, y, z).")
    exit(1)

# Sprawdzenie, czy liczba punktów jest zgodna z oczekiwaniami
if points.shape[0] != num_points:
    print(f"Error: points file must contain exactly {num_points} points.")
    exit(1)

# Tworzenie siatki punktów
grid = pv.PolyData(points)

# Tworzenie połączeń między punktami wzdłuż wysokości i obwodu
lines = []
for i in range(num_height_points - 1):
    for j in range(num_circumferential_points):
        next_j = (j + 1) % num_circumferential_points
        # Łączenie punktów wzdłuż wysokości
        lines.append(
            [
                i * num_circumferential_points + j,
                (i + 1) * num_circumferential_points + j,
            ]
        )
        # Łączenie punktów wzdłuż obwodu
        lines.append(
            [
                i * num_circumferential_points + j,
                i * num_circumferential_points + next_j,
            ]
        )

# Tworzenie struktury linii
lines = np.array(lines)
num_lines = lines.shape[0]
cells = np.hstack([np.full((num_lines, 1), 2), lines]).flatten()

# Dodawanie linii do siatki
grid.lines = cells

# Tworzenie powierzchni kafelków
faces = []
centers = []
for i in range(num_height_points - 1):
    for j in range(num_circumferential_points):
        next_j = (j + 1) % num_circumferential_points
        p1 = points[i * num_circumferential_points + j]
        p2 = points[i * num_circumferential_points + next_j]
        p3 = points[(i + 1) * num_circumferential_points + next_j]
        p4 = points[(i + 1) * num_circumferential_points + j]
        faces.append(
            [
                4,
                i * num_circumferential_points + j,
                i * num_circumferential_points + next_j,
                (i + 1) * num_circumferential_points + next_j,
                (i + 1) * num_circumferential_points + j,
            ]
        )
        center = (p1 + p2 + p3 + p4) / 4
        centers.append(center)

faces = np.hstack(faces)
surface = pv.PolyData(points, faces)

# Obliczenie całkowitej powierzchni kafelków
total_tile_area = tile_area * num_points

# Objętość sfery i obliczenie promienia
t = 4e-3  # 4 mm
sphere_volume = tile_area * t
sphere_radius = (3 * sphere_volume / (4 * np.pi)) ** (1 / 3)

# Tworzenie sfer w centrach kafelków
# spheres = pv.PolyData()
# for center in centers:
#     sphere = pv.Sphere(radius=sphere_radius, center=center)
#     spheres += sphere
# Porównanie powierzchni
print(f"Powierzchnia boczna walca: {cylinder_surface_area}")
print(f"Suma powierzchni kafelków: {total_tile_area}")

# Sprawdzenie, czy powierzchnie są równe
if np.isclose(cylinder_surface_area, total_tile_area):
    print("Suma powierzchni kafelków jest równa powierzchni bocznej walca.")
else:
    print("Suma powierzchni kafelków NIE jest równa powierzchni bocznej walca.")

# Wizualizacja w PyVista
plotter = pv.Plotter()
plotter.add_mesh(surface, color="red", opacity=0.5, show_edges=True)
plotter.add_mesh(grid, color="blue", point_size=5, render_points_as_spheres=True)
# plotter.add_mesh(spheres, color="yellow", opacity=0.8)
plotter.add_mesh(np.array(centers), color="red")
plotter.show()
