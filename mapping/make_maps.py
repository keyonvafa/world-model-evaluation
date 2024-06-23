import folium
import pickle
import osmnx as ox
import networkx as nx
import os
import numpy as np


def create_curved_edge(lat1, lon1, lat2, lon2, direction, num_points=50):
    # Calculate the control point for the quadratic Bezier curve
    delta_lat = 0
    delta_lon = 0
    if "N" in direction:
        delta_lat = 0.002
    elif "S" in direction:
        delta_lat = -0.002
    if "E" in direction:
        delta_lon = 0.002
    elif "W" in direction:
        delta_lon = -0.002

    control_lat = lat1 + delta_lat
    control_lon = lon1 + delta_lon

    # Generate points along the Bezier curve
    t = np.linspace(0, 1, num_points)
    lat_curve = (1 - t) ** 2 * lat1 + 2 * (1 - t) * t * control_lat + t**2 * lat2
    lon_curve = (1 - t) ** 2 * lon1 + 2 * (1 - t) * t * control_lon + t**2 * lon2

    # Create a list of coordinate tuples for the curved edge
    edge_coordinates = list(zip(lat_curve, lon_curve))
    return edge_coordinates


# def white_bg_tile(x, y, z):
# return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQAAAAEAAQMAAABmvDolAAAAA1BMVEUAAACnej3aAAAAAXRSTlMAQObYZgAAACJJREFUaIHtwTEBAAAAwqD1T20ND6AAAAAAAAAAAAAA4N8AKvgAAUFIrrEAAAAASUVORK5CYII='


def make_map(graph, lat_long_to_intersection_id, far):
    # Create a Folium map centered on Midtown
    tile = None
    # tile = folium.raster_layers.TileLayer(opacity=0.2)
    if far:
        lat_range = (40.702, 40.800)
        long_range = (-74.022, -73.933)
        zoom = 13
        weight = 0.75
        radius = 1
        cmap_new = ["black", "black"]
        cmap_true = ["black", "black"]
        alpha = 0.3
        # tile = None
    else:
        lat_range = (40.714, 40.72)
        long_range = (-74.02, -73.98)
        zoom = 16
        weight = 1.5
        radius = 2
        cmap_new = ["lightsalmon", "firebrick"]
        cmap_true = ["black", "black"]
        alpha = 0.7
        # tile = None

    center = (sum(lat_range) / 2, sum(long_range) / 2)
    map_nyc = folium.Map(
        location=center, zoom_start=zoom, tiles=tile, zoom_control=False
    )

    # Add edges as lines to the map with labels
    for u, v, k, data in graph.edges(keys=True, data=True):
        lat1, long1 = intersection_id_to_lat_long[u]
        lat2, long2 = intersection_id_to_lat_long[v]
        etype = data["edge_type"]
        direction = data["direction"]
        if etype == "true_unused":
            continue
        elif etype == "true":
            edge_locations = [[lat1, long1], [lat2, long2]]
            edge_line = folium.ColorLine(
                positions=edge_locations,
                weight=1,
                colors=np.arange(len(edge_locations) - 1),
                colormap=cmap_true,
                alpha=0.3,
            )
        else:
            curved_edge_locations = create_curved_edge(
                lat1, long1, lat2, long2, direction, num_points=50
            )
            edge_line = folium.ColorLine(
                positions=curved_edge_locations,
                weight=weight,
                colors=np.arange(len(curved_edge_locations) - 1),
                colormap=cmap_new,
                alpha=alpha,
            )

        edge_line.add_to(map_nyc)
    return map_nyc


if __name__ == "__main__":
    place_name = "Manhattan, New York City, New York, USA"
    ox_graph = ox.graph_from_place(place_name, network_type="drive")
    nodes = list(ox_graph.nodes())

    intersection_id_to_lat_long = {}
    for node in nodes:
        intersection_id_to_lat_long[node] = (
            ox_graph.nodes[node]["y"],
            ox_graph.nodes[node]["x"],
        )
    lat_long_to_intersection_id = {v: k for k, v in intersection_id_to_lat_long.items()}

    graph_names = [x for x in os.listdir("graphs/") if x[0] != "."]
    for graph_name in graph_names:
        with open("graphs/" + graph_name, "rb") as f:
            graph = pickle.load(f)
        print(graph_name)
        print(graph)
        far_map = make_map(graph, lat_long_to_intersection_id, True)
        close_map = make_map(graph, lat_long_to_intersection_id, False)
        far_map.save("maps/far_{}.html".format(graph_name[:-4]))
        close_map.save("maps/close_{}.html".format(graph_name[:-4]))
