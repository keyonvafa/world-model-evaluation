import networkx as nx
import copy
from collections import defaultdict
import utils


def get_used_degree(graph, node):
    """Return the degree of a given node ignoring true_unused edges"""
    ctr = 0
    for u, v, k, etype in graph.out_edges(node, data="edge_type", keys=True):
        if etype != "true_unused":
            ctr += 1
    return ctr


def check_node_using_direction(graph, node, direction):
    direction_exists = False
    for u, v, k, edge_data in graph.out_edges(node, data=True, keys=True):
        if (
            edge_data["edge_type"] != "true_unused"
            and edge_data["direction"] == direction
        ):
            assert not direction_exists  # only should happen once
            direction_exists = True
    return direction_exists


def near_neighbors(graph, cur_node, node_to_lat_long, radius):
    """Return nearby neighbors in Manhattan distance of lat/long"""
    cur_lat_long = node_to_lat_long[cur_node]
    return [
        node
        for node in graph.nodes
        if utils.distance_lat_long_to_miles(cur_lat_long, node_to_lat_long[node])
        <= radius
    ]


def bfs_to_failure(
    graph, source, dest, directions, max_degree, initial_node_and_direction=None
):
    """
    Check if a list of directions successfully traverses from source to destination.
    If not, return the longest valid subpath that ends at a node with fewer than
    max_degree neighbors and no edge of the next direction (so that we could add a new edge).
    """
    # BFS queue of (cur_node, path, degree_inc) entries
    # cur_node: is a value indicating current head of the path
    # path: list of edges where each edge is a (u,v,k) tuple
    # new_directions: dictionary of newly used node: directions pairs
    used_directions = defaultdict(list)
    if initial_node_and_direction:
        node, direction = initial_node_and_direction
        used_directions[node].append(direction)
    queue = [(source, [], used_directions)]
    longest_valid_path = []  # keep track of longest valid path
    # BFS on valid hops to see if a valid path already exists
    while len(queue) > 0:
        cur_node, path, used_directions = queue.pop(0)
        next_direction = directions[len(path)]
        assert (
            len(used_directions[cur_node]) + get_used_degree(graph, cur_node)
            <= max_degree
        )
        # Update along valid neighbors
        for u, v, k, edge_dict in graph.out_edges(cur_node, data=True, keys=True):
            edge = (u, v, k)
            direction = edge_dict["direction"]
            etype = edge_dict["edge_type"]
            # Check valid edge corresponding to next direction
            if direction == next_direction:
                new_used_directions = used_directions
                # Check if path would use a previously usused true edge
                if etype == "true_unused":
                    if (
                        initial_node_and_direction
                        and u == initial_node_and_direction[0]
                        and next_direction == initial_node_and_direction[1]
                    ):
                        continue
                    elif next_direction not in used_directions[u]:
                        # Check if using this edge pushes degree over maximum
                        if (
                            len(used_directions[u]) + get_used_degree(graph, u)
                            == max_degree
                        ):
                            continue
                        # Check if we are already added a new edge with this direction
                        elif check_node_using_direction(graph, u, direction):
                            continue
                        # If both checks pass, we can use this edge
                        else:
                            new_used_directions = copy.deepcopy(used_directions)
                            new_used_directions[u].append(next_direction)

                new_path = copy.deepcopy(path) + [edge]
                # Check if full path reaches destination
                if len(new_path) == len(directions):
                    if v == dest:
                        return new_path
                    continue  # invalid path, go to next neighbor

                # Add queue element
                queue.append((v, new_path, new_used_directions))
                # Update longest path only if the final node has room for an extra edge
                # in terms of max degree and direction
                next_next_direction = directions[len(new_path)]
                if (
                    len(new_used_directions[v]) + get_used_degree(graph, v) < max_degree
                    and not check_node_using_direction(graph, v, next_next_direction)
                    and not next_next_direction in new_used_directions[v]
                    and len(new_path) > len(longest_valid_path)
                ):
                    longest_valid_path = new_path

    return longest_valid_path


def reconstruct_sequence(graph, sequence, valid_new_nbr_fn, max_degree, loss_fn=None):
    """Take in sequence as list of source, dest, directions"""
    source, dest = sequence[:2]
    directions = sequence[2:]

    path = bfs_to_failure(graph, source, dest, directions, max_degree)
    # Update edge usage information
    for edge in path:
        if graph.edges[edge]["edge_type"] == "true_unused":
            assert not check_node_using_direction(
                graph, edge[0], graph.edges[edge]["direction"]
            )
            graph.edges[edge]["edge_type"] = "true"

    # Repeatedly bfs_to_failure, adding in new edges to continue
    while len(path) < len(directions):
        next_direction = directions[len(path)]

        if len(path) > 0:
            cur_node = path[-1][1]
        else:
            cur_node = source
            # If source has too many edges, fail for this sequence
            if get_used_degree(
                graph, cur_node
            ) >= max_degree or check_node_using_direction(
                graph, cur_node, next_direction
            ):
                return False

        # Special case of one extra edge needed, add shortcut to dest
        if len(path) == len(directions) - 1:
            # If destination is too far away, fail
            if dest not in valid_new_nbr_fn(graph, cur_node):
                return False
            key = graph.add_edge(
                cur_node, dest, direction=next_direction, edge_type="new"
            )
            assert get_used_degree(graph, cur_node) <= max_degree
            path += [(cur_node, dest, key)]
            break

        # Try starting BFS from all nearby neighbors
        best_subpath = []
        for node in valid_new_nbr_fn(graph, cur_node):
            subpath = bfs_to_failure(
                graph,
                node,
                dest,
                directions[len(path) + 1 :],
                max_degree,
                initial_node_and_direction=(cur_node, next_direction),
            )
            if len(subpath) > len(best_subpath):
                best_subpath = subpath

        if len(best_subpath) > 0:
            # add edge from current node to start of longest continuation
            assert not check_node_using_direction(graph, cur_node, next_direction)
            assert best_subpath[0][0] in valid_new_nbr_fn(graph, cur_node)
            key = graph.add_edge(
                cur_node, best_subpath[0][0], direction=next_direction, edge_type="new"
            )
            assert get_used_degree(graph, cur_node) <= max_degree
            # update edge usage information
            for edge in best_subpath:
                if graph.edges[edge]["edge_type"] == "true_unused":
                    assert not check_node_using_direction(
                        graph, edge[0], graph.edges[edge]["direction"]
                    )
                    graph.edges[edge]["edge_type"] = "true"
            # update path
            path += [(cur_node, best_subpath[0][0], key)] + best_subpath
        else:  # couldn't find any way to make progress, give up on sequence
            return False

    # assertion to check path is satisfied
    prev_node = source
    for i, edge in enumerate(path):
        assert edge[0] == prev_node
        assert graph.edges[edge]["direction"] == directions[i]
        prev_node = edge[1]
    assert path[-1][1] == dest

    return True


def reconstruct_graph(
    true_graph, sequences, node_to_lat_long, max_degree=4, max_distance=0.5
):
    reconst_graph = true_graph.copy()
    for edge in reconst_graph.edges:
        reconst_graph.edges[edge]["edge_type"] = "true_unused"

    failed_ctr = 0
    reasonable_nbr_fn = lambda G, u: near_neighbors(
        G, u, node_to_lat_long, max_distance
    )
    for i, sequence in enumerate(sequences):
        # if (i + 1) % 1000 == 0:
        #     print(i + 1)
        sequence_success = reconstruct_sequence(
            reconst_graph, sequence, reasonable_nbr_fn, max_degree=max_degree
        )
        if not sequence_success:
            failed_ctr += 1

    print("# Sequences Failed / # Sequences Total")
    print(failed_ctr, "/", len(sequences))

    # Check max_degree
    for node in reconst_graph.nodes:
        assert get_used_degree(reconst_graph, node) <= max_degree

    return reconst_graph
