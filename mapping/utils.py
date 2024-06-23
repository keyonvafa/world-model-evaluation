import networkx as nx
from datetime import datetime
import math


def distance_lat_long_to_miles(latlong1, latlong2):
    lat1 = latlong1[0]
    lon1 = latlong1[1]
    lat2 = latlong2[0]
    lon2 = latlong2[1]
    # Radius of the Earth in miles
    radius = 3958.8

    # Convert degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = radius * c

    return distance


# def validate_turns_sequence(
#     node_and_direction_to_neighbor, valid_turns, sequence, verbose=True
# ):
#     source, dest = sequence[:2]
#     directions = sequence[2:]
#     cur_node = source
#     for i, direction in enumerate(directions):
#         if direction in valid_turns[cur_node]:
#             cur_node = node_and_direction_to_neighbor[(cur_node, direction)]
#         else:
#             if verbose:
#                 print(
#                     "Path Failure (turns): ", source, dest, i, directions[i], directions
#                 )
#             return False
#     if cur_node != dest:
#         if verbose:
#             print("Dest Failure (turns)", source, dest, directions)
#         return False
#     return True


def validate_graph_sequences(graph, sequences, verbose=True):
    successes = []
    path_failures = []
    dest_failures = []
    for sequence in sequences:
        source, dest = sequence[:2]
        directions = sequence[2:]
        cur_node = source
        failure = False
        for i, direction in enumerate(directions):
            next_node = -1
            for u, v, k, d in graph.out_edges(cur_node, data="direction", keys=True):
                if d == direction:
                    next_node = v
                    break
            if next_node == -1:
                path_failures.append(sequence)
                failure = True
                break
            cur_node = next_node
        if failure:
            continue
        if cur_node != dest:
            dest_failures.append(sequence)
        else:
            successes.append(sequence)
    if verbose:
        print("# Sequences:", len(sequences))
        print(
            "# Successes: {} ({:.1f}%)".format(
                len(successes), len(successes) / len(sequences) * 100
            )
        )
        print(
            "# Path Failures: {} ({:.1f}%)".format(
                len(path_failures), len(path_failures) / len(sequences) * 100
            )
        )
        print(
            "# Dest Failures: {} ({:.1f}%)".format(
                len(dest_failures), len(dest_failures) / len(sequences) * 100
            )
        )
    return successes, path_failures, dest_failures


def sample2sequence(sample, verbose=False):
    tokens = sample.split(" ")
    if len(tokens) <= 3 or tokens[-1] != "end":
        if verbose:
            print("Ignoring sample ", tokens)
        return []
    source = int(tokens[0])
    dest = int(tokens[1])
    return [source, dest] + tokens[2:-1]


def get_timestamp():
    timestamp = datetime.now()
    # Format the timestamp as a string
    return timestamp.strftime("%Y-%m-%d_%H-%M-%S")


def annotate_error_types(graph):
    """Given a graph with true and new edges, annotate what type of failures the new edges are"""
    pass
