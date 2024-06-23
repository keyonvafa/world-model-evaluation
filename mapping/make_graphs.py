import networkx as nx
import pickle
import osmnx as ox
import utils
import reconstruction
import numpy as np
import argparse


def parse_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process command line arguments")

    # Add arguments
    parser.add_argument(
        "--dataset", type=str, default="shortest-paths", help="Dataset to run on"
    )
    parser.add_argument(
        "--degree", type=int, default=4, help="Maximum out-degree of each node"
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=0.5,
        help="Maximum distance (in miles) for adding new edges",
    )
    parser.add_argument(
        "--nsequences",
        type=int,
        default=1000,
        help="Number of sequences to run reconstruction on",
    )
    parser.add_argument(
        "--randomerr",
        default=-1,
        type=float,
        help="Take only good sequences, and then randomly corrupt a given fraction of them.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Access the argument values
    return args.dataset, args.nsequences, args.degree, args.distance, args.randomerr


def build_true_graph(nodes, node_to_lat_long, node_and_direction_to_neighbor):
    true_graph = nx.MultiDiGraph()
    for node in nodes:
        true_graph.add_node(node, lat_long=node_to_lat_long[node])
    for node, direction in node_and_direction_to_neighbor:
        nbr = node_and_direction_to_neighbor[(node, direction)]
        assert not true_graph.has_edge(node, nbr)
        true_graph.add_edge(node, nbr, direction=direction, edge_type="true_unused")
    return true_graph


def get_nodes_and_lat_longs():
    place_name = "Manhattan, New York City, New York, USA"
    ox_graph = ox.graph_from_place(place_name, network_type="drive")
    nodes = list(ox_graph.nodes())
    node_to_lat_long = {}
    for node in nodes:
        node_to_lat_long[node] = (
            ox_graph.nodes[node]["y"],
            ox_graph.nodes[node]["x"],
        )
    return nodes, node_to_lat_long


def load_manhattan(folder, do_check=True):
    with open("sequences/" + folder + "/node_and_direction_to_neighbor.pkl", "rb") as f:
        node_and_direction_to_neighbor = pickle.load(f)
    with open("sequences/" + folder + "/valid_turns.pkl", "rb") as f:
        valid_turns = pickle.load(f)
        nodes, node_to_lat_long = get_nodes_and_lat_longs()

    bad_keys = []
    for (node, direction), nbr in node_and_direction_to_neighbor.items():
        if node not in nodes or nbr not in nodes:
            bad_keys.append((node, direction))
            print("Removing edge outside of nyc", node, direction, nbr)
    for key in bad_keys:
        del node_and_direction_to_neighbor[key]

    true_graph = build_true_graph(
        nodes, node_to_lat_long, node_and_direction_to_neighbor
    )

    # sanity check
    if do_check:
        print("Running graph build sanity check")
        for node in valid_turns:
            directions = valid_turns[node]
            assert (
                (node, direction) in node_and_direction_to_neighbor
                for direction in directions
            )
        for node, direction in node_and_direction_to_neighbor:
            assert direction in valid_turns[node]
        for node in true_graph.nodes:
            for u, v, k, d in true_graph.out_edges(node, data="direction", keys=True):
                assert d in valid_turns[node]

    return (
        nodes,
        node_to_lat_long,
        true_graph,
        node_and_direction_to_neighbor,
    )


def load_sequences(folder, nodes):
    with open("sequences/" + folder + "/samples.txt", "r") as f:
        samples = f.read().split("\n")
    print("# Samples", len(samples))

    sequences = [
        x for x in [utils.sample2sequence(sample) for sample in samples] if len(x) > 0
    ]
    print("# Ill-formed sequences", len(samples) - len(sequences))
    sequences = [x for x in sequences if x[0] in nodes and x[1] in nodes]

    return sequences


def pipeline(dataset, nsequences, max_degree, max_distance, randomerr):
    print("Loading Manhattan")
    nodes, node_to_lat_long, true_graph, node_and_direction_to_neighbor = (
        load_manhattan(dataset)
    )
    print("Loading sequences")
    sequences = load_sequences(dataset, nodes)
    np.random.shuffle(sequences)
    sequences = sequences[: min(len(sequences), nsequences)]
    # Validate successes, failures
    if randomerr < 0:
        successes, path_failures, dest_failures = utils.validate_graph_sequences(
            true_graph, sequences, verbose=True
        )
    else:
        successes, path_failures, dest_failures = utils.validate_graph_sequences(
            true_graph, sequences, verbose=False
        )
        possible_directions = ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]
        assert randomerr < 1
        sequences = successes
        errctr = 0
        for sequence in successes:
            if np.random.rand() < randomerr:
                idx = np.random.randint(2, len(sequence))
                sequence[idx] = np.random.choice(
                    [x for x in possible_directions if x != sequence[idx]]
                )
                errctr += 1
        successes, path_failures, dest_failures = utils.validate_graph_sequences(
            true_graph, sequences, verbose=True
        )
    print("Reconstructing Graph")
    reconst_graph = reconstruction.reconstruct_graph(
        true_graph, sequences, node_to_lat_long, max_degree, max_distance
    )
    # Count edge types
    n_new = 0
    n_true = 0
    n_unused = 0
    for node in reconst_graph.nodes:
        for u, v, k, d in reconst_graph.out_edges(node, keys=True, data=True):
            if d["edge_type"] == "new":
                n_new += 1
            elif d["edge_type"] == "true":
                n_true += 1
            elif d["edge_type"] == "true_unused":
                n_unused += 1
    print("New edges, True edges, Unused true edges")
    print(n_new, n_true, n_unused)
    return reconst_graph


if __name__ == "__main__":
    dataset, nsequences, max_degree, max_distance, randomerr = parse_args()
    identifier = "{}_seq{}_deg{}_dist{}_randerr{}".format(
        dataset, nsequences, max_degree, max_distance, randomerr
    )
    print(identifier)

    outfile = "graphs/" + identifier + ".pkl"
    reconst_graph = pipeline(dataset, nsequences, max_degree, max_distance, randomerr)
    print("Saving graph to ", outfile)
    with open(outfile, "wb") as f:
        pickle.dump(reconst_graph, f)
