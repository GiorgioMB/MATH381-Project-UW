import math
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.colors import Normalize
from shapely.geometry import Point, LineString


def verify_data():
    """
    Loads and processes road network data from a GeoJSON file.

    - Reads the dataset and ensures it uses a projected coordinate system.
    - Checks for the presence of the 'ADT' (Average Daily Traffic) column.
    - Computes busyness values and normalizes them for comparison.
    - Extracts or calculates road lengths.
    - Computes the average road length for reference.
    """
    # Load road network data
    data_path = "SDOT_data.geojson"
    roads = gpd.read_file(data_path)

    # Convert to projected coordinate system if necessary (for accurate distance calculations)
    if roads.crs.is_geographic:
        roads = roads.to_crs(epsg=26910)

    # Ensure 'ADT' column exists, otherwise raise an error
    if 'ADT' not in roads.columns:
        raise ValueError("Dataset is missing the ADT column required for busyness.")

    # Assign busyness values based on ADT
    roads['busyness'] = roads['ADT']
    roads['frequency'] = 1.0

    # Normalize busyness values for comparison
    roads['busyness_norm'] = (roads['busyness'] - roads['busyness'].min()) / (
            roads['busyness'].max() - roads['busyness'].min() + 1e-6)

    # Use existing road lengths or compute from geometry
    if 'Shape_Leng' in roads.columns:
        roads['length'] = roads['Shape_Leng']
    else:
        roads['length'] = roads.geometry.length

    # Compute average road length for reference
    avg_length = roads['length'].mean()

    return roads, avg_length


def extract_lines(geom):
    """
    Extract and yield individual LineStrings from a given geometry.

    This function handles both LineString and MultiLineString geometries.
    If the input is a LineString, it yields it directly.
    If the input is a MultiLineString, it iterates over its components and yields each LineString separately.
    """
    # Check if the geometry is a single LineString
    if geom.geom_type == 'LineString':
        yield geom

    # If it's a MultiLineString, iterate through its components
    elif geom.geom_type == 'MultiLineString':
        for line in geom.geoms:
            yield line


def vertex_key(coord, precision=3):
    """
    Generate a unique key for a vertex by rounding its coordinates.

    This function rounds the x and y coordinates of a given point to the specified precision. It is useful for spatial indexing or reducing floating-point precision errors.
    """
    return round(coord[0], precision), round(coord[1], precision)


def make_graph(roads, avg_length):
    """
    Construct a road network graph from given road geometries and compute hub scores.

    This function takes a set of road geometries and constructs a graph where each road segment is represented as an edge between two nodes (start and end coordinates). It assigns attributes 
    such as length, busyness, and geometry to each edge and computes hub scores for nodes based on incident edge busyness. The function also normalizes hub scores and assigns them as node attributes. 
    
    Finally, it computes edge weights using a custom function and returns the minimum spanning tree of the graph, along with hub scores and the full graph.
    """
    # Create an empty graph
    G = nx.Graph()

    # Build graph from road network
    for idx, row in roads.iterrows():
        geom = row.geometry
        for line in extract_lines(geom):
            start = vertex_key(line.coords[0])
            end = vertex_key(line.coords[-1])

            # Edge attributes: length, busyness, and geometry
            attr = {
                'length': line.length,
                'busyness_norm': row['busyness_norm'],
                'geometry': line
            }

            # Add edge to graph, incrementing count if it already exists
            if G.has_edge(start, end):
                G[start][end]['count'] = G[start][end].get('count', 1) + 1
            else:
                G.add_edge(start, end, **attr)

    # Compute hub scores for each node (sum of incident edge busyness)
    hub_scores = {}
    for node in G.nodes():
        incident = G.edges(node, data=True)
        total = sum(data.get('busyness_norm', 0) for _, _, data in incident)
        hub_scores[node] = total

    # Normalize hub scores
    max_hub = max(hub_scores.values()) if hub_scores else 1.0
    for node in hub_scores:
        hub_scores[node] /= (max_hub + 1e-6)

    # Assign hub scores as node attributes
    nx.set_node_attributes(G, hub_scores, 'hub_score')
    for u, v, data in G.edges(data=True):
        start_hub = hub_scores.get(u, 0)
        end_hub = hub_scores.get(v, 0)
        data['weight'] = compute_edge_weight(data, start_hub, end_hub, avg_length,
                                             lambda_m=1.0, lambda_c=1.0, beta=1.0,
                                             base_freq=5, scale=10, gamma=0.05)

    mst = nx.minimum_spanning_tree(G, weight='weight')
    return mst, hub_scores, G


def compute_edge_weight(edge_data, start_hub, end_hub, avg_length,
                        lambda_m=1.0, lambda_c=1.0, beta=2.0,
                        base_freq=5, scale=10, gamma=0.2):
    """
    Compute a composite weight for an edge in a transportation network.

    This function calculates the weight of an edge based on multiple factors, including road length, busyness mismatch, convenience penalties for highly busy hubs, bonuses for connecting busy hubs, estimated transit frequency, and fare-related costs. 

    The weight is designed to optimize routing decisions in a transportation network by considering factors that impact travel efficiency and user experience.
    """
    # Extract the length of the edge
    length = edge_data['length']

    # Compute mismatch penalty: penalizes roads with busyness far from 1.0
    mismatch = abs(edge_data['busyness_norm'] - 1.0)
    mismatch_penalty = lambda_m * mismatch

    # Define a threshold for considering a node as a busy hub
    busy_threshold = 0.7

    # Apply a convenience penalty if both endpoints are highly busy hubs
    if (start_hub > busy_threshold) and (end_hub > busy_threshold):
        convenience_penalty = lambda_c * (length / avg_length)
    else:
        convenience_penalty = 0

    # Compute a bonus for edges that connect high-busyness hubs
    hub_bonus = beta * (start_hub + end_hub)

    # Compute an average factor for estimating transit frequency
    avg_factor = (edge_data['busyness_norm'] + start_hub + end_hub) / 3.0

    # Estimate frequency of transit service on this edge
    freq_est = base_freq + scale * avg_factor

    # Compute a fare-related cost, proportional to estimated frequency
    fare_cost = gamma * freq_est

    # Final weight calculation: accounts for length, penalties, bonuses, and fare cost
    weight = length * (1 + mismatch_penalty + convenience_penalty) - hub_bonus * avg_length * 0.1 + fare_cost
    return max(weight, 0.1)


def assign_frequency(edge_data, start_hub, end_hub, base_freq=5, scale=10):
    """
    Estimate the transit service frequency for a given road segment.

    The function calculates the expected frequency of transit services on a road segment based on its busyness and the hub scores of its endpoints. Higher busyness and hub scores lead to higher estimated frequency.
    """
    # Compute an average factor based on road busyness and hub scores
    avg_factor = (edge_data['busyness_norm'] + start_hub + end_hub) / 3.0

    # Calculate the frequency based on the base frequency and scaled busyness
    freq = base_freq + scale * avg_factor

    return freq


def get_endpoint_coords(route, G):
    """
    Returns the coordinates of the start and end nodes of the route.
    Assumes that nodes in G have a 'x' and 'y' attribute or are coordinate tuples.
    """

    def get_coord(node):
        if isinstance(node, tuple) and len(node) == 2:
            return node
        node_data = G.nodes[node]
        return node_data.get('x', 0), node_data.get('y', 0)

    return get_coord(route[0]), get_coord(route[-1])


def euclidean_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_geometric_penalty(route, close_threshold=10):
    """
    Compute a penalty that increases if the route deviates from forming a simple geometric figure.
    
    Components:
      - Openness penalty: if the route is not closed (start and end are far apart),
        a penalty proportional to that distance is applied.
      - Turning penalty: measures the average absolute turning angle (in radians) at internal vertices.
        Many or erratic turns suggest that extra edges are forcing the route away from a simple shape.
    """
    pts = [np.array(p) for p in route]
    openness_penalty = 0.0
    if euclidean_dist(route[0], route[-1]) > close_threshold:
        openness_penalty = euclidean_dist(route[0], route[-1])

    total_turn = 0.0
    count = 0
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            continue
        angle = np.arccos(np.clip(np.dot(v1, v2) / norm_product, -1, 1))
        total_turn += abs(angle)
        count += 1
    turning_penalty = (total_turn / count) if count > 0 else 0.0

    return openness_penalty + turning_penalty


def merge_two_routes_improved(route1, route2, G, forbidden_penalty=100.0,
                              merge_length_factor=30, geometric_factor=1.0):
    """
    Attempt to merge two routes with additional penalties:
    
    - Penalizes merging longer routes (to favor merging smaller ones).
    - Penalizes routes that deviate from forming a simple geometric figure.
      This penalty increases if the route is not closed or has erratic turning angles.
    - Applies an extra cost for traversing 'forbidden' roads (roads not in the original MST).
    
    Parameters:
      - route1, route2: lists of nodes (as coordinate tuples or nodes with 'x','y').
      - G: networkx graph with edge attributes. If an edge is not in the original MST,
           it should have attribute 'forbidden'=True.
      - forbidden_penalty: extra cost added for each forbidden edge in the connecting path.
      - merge_length_factor: multiplier for the combined physical route length.
      - geometric_factor: multiplier for the geometric (shape) penalty.
      
    Returns:
      - best_merge: the merged route (list of nodes) if a merge is found.
      - best_total_cost: the total cost for the merge.
    """
    best_total_cost = math.inf
    best_merge = None

    def custom_weight(u, v, d):
        base = d.get('length', 0)
        if d.get('forbidden', False):
            base += forbidden_penalty
        return base

    def compute_route_length(route):
        total = 0.0
        for i in range(len(route) - 1):
            total += euclidean_dist(route[i], route[i + 1])
        return total

    options = [
        (route1, route2, route1[-1], route2[0]),
        (route1, list(reversed(route2)), route1[-1], list(reversed(route2))[0]),
        (list(reversed(route1)), route2, list(reversed(route1))[-1], route2[0]),
        (list(reversed(route1)), list(reversed(route2)), list(reversed(route1))[-1], list(reversed(route2))[0])
    ]

    for r1, r2, u, v in options:
        try:
            path = nx.shortest_path(G, source=u, target=v, weight=custom_weight)
            spath_cost = nx.shortest_path_length(G, source=u, target=v, weight=custom_weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        merged_route = r1 + path[1:] + r2
        route_length = compute_route_length(merged_route)
        geom_pen = compute_geometric_penalty(merged_route)

        total_cost = spath_cost + merge_length_factor * route_length + geometric_factor * geom_pen

        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_merge = merged_route

    return best_merge, best_total_cost

def optimize_transit(mst, hub_scores, G, roads):
    for u, v, data in mst.edges(data=True):
        start_hub = hub_scores.get(u, 0)
        end_hub = hub_scores.get(v, 0)
        freq = assign_frequency(data, start_hub, end_hub)
        data['service_frequency'] = freq
        data['avg_wait_time'] = 60.0 / freq

    redundancy_threshold = 0.9
    redundant_edges = []
    for u, v, data in G.edges(data=True):
        if mst.has_edge(u, v):
            continue
        if hub_scores.get(u, 0) > redundancy_threshold and hub_scores.get(v, 0) > redundancy_threshold:
            redundant_edges.append((u, v, data))

    augmented_network = mst.copy()
    for u, v, data in redundant_edges:
        start_hub = hub_scores.get(u, 0)
        end_hub = hub_scores.get(v, 0)
        freq = assign_frequency(data, start_hub, end_hub)
        data['service_frequency'] = freq
        data['avg_wait_time'] = 60.0 / freq
        augmented_network.add_edge(u, v, **data)

    aug_edges = []
    wait_times = []
    for u, v, data in augmented_network.edges(data=True):
        geom = data.get('geometry')
        if not isinstance(geom, LineString):
            geom = LineString([Point(u), Point(v)])
        aug_edges.append(geom)
        wait_times.append(data.get('avg_wait_time', 60))

    aug_gdf = gpd.GeoDataFrame(geometry=aug_edges, crs=roads.crs)

    norm = Normalize(vmin=min(wait_times), vmax=max(wait_times))
    cmap = cm.get_cmap('coolwarm')

    fig, ax = plt.subplots(figsize=(12, 12))
    roads.plot(ax=ax, color="black", linewidth=1, label="Original Road Network")

    for geom, wt in zip(aug_edges, wait_times):
        ax.plot(*geom.xy, color=cmap(norm(wt)), linewidth=1.2)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Average Wait Time (min)")

    ax.set_title("MST Edges with Average Wait Time")
    ax.legend()
    plt.savefig("heatmap_network.png")
    plt.show()
    return augmented_network

def reduce_route_count(routes, G, target_count=600, dist_threshold=110, max_iterations=2000,
                       forbidden_penalty=100.0, merge_length_factor=30, geometric_factor=1.0):
    """
    Iteratively merge pairs of routes until the total number is less than or equal to target_count.
    Uses additional penalties to favor merging smaller routes and routes that can be interpreted
    as simple geometric figures.
    """
    routes = routes.copy()
    iteration = 0
    while len(routes) > target_count and iteration < max_iterations:
        best_total_cost = math.inf
        best_pair = None
        best_merged = None

        endpoints = [get_endpoint_coords(route, G) for route in routes]

        for i in range(len(routes)):
            start_i, end_i = endpoints[i]
            for j in range(i + 1, len(routes)):
                start_j, end_j = endpoints[j]
                if (euclidean_dist(end_i, start_j) < dist_threshold or
                        euclidean_dist(end_i, end_j) < dist_threshold or
                        euclidean_dist(start_i, start_j) < dist_threshold or
                        euclidean_dist(start_i, end_j) < dist_threshold):

                    merged, cost = merge_two_routes_improved(
                        routes[i], routes[j], G,
                        forbidden_penalty=forbidden_penalty,
                        merge_length_factor=merge_length_factor,
                        geometric_factor=geometric_factor
                    )
                    if merged is not None and cost < best_total_cost:
                        best_total_cost = cost
                        best_pair = (i, j)
                        best_merged = merged

        if best_pair is None:
            break

        i, j = best_pair
        new_routes = [routes[k] for k in range(len(routes)) if k not in best_pair]
        new_routes.append(best_merged)
        routes = new_routes
        iteration += 1
        print(f"Iteration {iteration}: Merged routes {i} and {j}; Total routes now = {len(routes)}")
    return routes


def prepare_edge_demand(G):
    edge_demand = {}
    for u, v, data in G.edges(data=True):
        demand = max(1, int(round(data.get('service_frequency', 1))))
        key = tuple(sorted((u, v)))
        edge_demand[key] = edge_demand.get(key, 0) + demand
    return edge_demand


def generate_bus_routes(G):
    edge_demand = prepare_edge_demand(G)
    routes = []
    while any(d > 0 for d in edge_demand.values()):
        candidate_edge, demand_val = max(edge_demand.items(), key=lambda item: item[1])
        if demand_val <= 0:
            break

        u, v = candidate_edge
        edge_demand[candidate_edge] -= 1
        route = [u, v]

        def extend_route(endpoint, front=True):
            current = endpoint
            while True:
                best_n = None
                best_demand = 0
                for neighbor in G.neighbors(current):
                    key = tuple(sorted((current, neighbor)))
                    if edge_demand.get(key, 0) > best_demand:
                        best_demand = edge_demand[key]
                        best_n = neighbor
                if best_n is None or best_demand <= 0:
                    break
                if front:
                    route.insert(0, best_n)
                else:
                    route.append(best_n)
                key = tuple(sorted((current, best_n)))
                edge_demand[key] -= 1
                current = best_n

        extend_route(route[0], front=True)
        extend_route(route[-1], front=False)
        routes.append(route)
    return routes


def merge_routes_simple(routes):
    merged = True
    while merged:
        merged = False
        new_routes = []
        used = [False] * len(routes)
        for i in range(len(routes)):
            if used[i]:
                continue
            merged_route = routes[i]
            used[i] = True
            for j in range(len(routes)):
                if used[j]:
                    continue
                if merged_route[-1] == routes[j][0]:
                    merged_route = merged_route + routes[j][1:]
                    used[j] = True
                    merged = True
            new_routes.append(merged_route)
        routes = new_routes
    return routes


def display_data(augmented_network):
    initial_bus_routes = generate_bus_routes(augmented_network)
    print("Initial number of routes:", len(initial_bus_routes))

    initial_bus_routes = merge_routes_simple(initial_bus_routes)
    print("Routes after simple merge:", len(initial_bus_routes))

    final_bus_routes = reduce_route_count(initial_bus_routes, augmented_network, target_count=600)
    final_bus_routes = reduce_route_count(final_bus_routes, augmented_network, target_count=600, dist_threshold=500)
    print("Final number of routes:", len(final_bus_routes))

    fig, ax = plt.subplots(figsize=(12, 12))
    for u, v, data in augmented_network.edges(data=True):
        geom = data.get('geometry')
        if not isinstance(geom, LineString):
            geom = LineString([Point(u), Point(v)])
        x, y = geom.xy
        ax.plot(x, y, color="gray", linewidth=1, zorder=1)

    colors = cm.get_cmap('tab20', len(final_bus_routes))
    for idx, route in enumerate(final_bus_routes):
        pts = [Point(n) if not isinstance(n, Point) else n for n in route]
        line = LineString(pts)
        x, y = line.xy
        ax.plot(x, y, color=colors(idx), linewidth=3, label=f"Route {idx + 1}", zorder=2)

    ax.set_title("Final Bus Routes")
    ax.legend()
    plt.savefig("final_routes.png")
    plt.show()

    edge_traces = []
    for u, v, data in augmented_network.edges(data=True):
        geom = data.get('geometry')
        if not isinstance(geom, LineString):
            geom = LineString([Point(u), Point(v)])
        x, y = geom.xy
        x = list(x)
        y = list(y)
        edge_traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                line=dict(color='gray', width=1),
                hoverinfo='none'
            )
        )

    cmap = cm.get_cmap('tab20', len(final_bus_routes))

    return final_bus_routes, cmap, edge_traces


def rgba_to_rgb_str(rgba):
    r, g, b, _ = rgba
    return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"


def display_bus_routes(final_bus_routes, cmap, edge_traces):
    route_traces = []
    for idx, route in enumerate(final_bus_routes):
        pts = [Point(n) if not isinstance(n, Point) else n for n in route]
        line = LineString(pts)
        x, y = line.xy
        x = list(x)
        y = list(y)
        route_traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                line=dict(color=rgba_to_rgb_str(cmap(idx)), width=3),
                name=f"Route {idx + 1}",
                visible=False
            )
        )

    all_traces = edge_traces + route_traces
    fig = go.Figure(data=all_traces)
    buttons = []

    visible_all = [True] * len(edge_traces) + [True] * len(route_traces)
    buttons.append(dict(
        label="All Routes",
        method="update",
        args=[{"visible": visible_all},
              {"title": "Final Bus Routes - All Routes"}]
    ))

    for i in range(len(final_bus_routes)):
        visible = [True] * len(edge_traces) + [False] * len(route_traces)
        visible[len(edge_traces) + i] = True
        buttons.append(dict(
            label=f"Route {i + 1}",
            method="update",
            args=[{"visible": visible},
                  {"title": f"Final Bus Routes - Route {i + 1}"}]
        ))

    fig.update_layout(
        title="Final Bus Routes",
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=1.1,
            y=1
        )],
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        template="plotly_white"
    )

    fig.write_html("bus_routes.html")
    fig.show()


def euclidean_distance(p1, p2):
    """Compute Euclidean distance between two 2D points (tuples)."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def generate_global_stop_candidates(routes, extra_stop_gap=500, merge_tolerance=20):
    """
    Given a list of routes (each a list of node coordinates),
    generate bus stop candidates that satisfy:
      1) Routes share stops if they pass through the same intersection.
      2) If the distance between two mandatory stops exceeds extra_stop_gap (meters), 
         extra stops are inserted along the route.
      3) Stops that are very near (within merge_tolerance) are merged across routes.
    Returns:
      global_stops: a list of global stop coordinates.
      route_stop_indices: a list (one per route) of indices into global_stops indicating
                          which stops are used by that route.
    """
    node_frequency = {}
    for route in routes:
        for pt in route:
            node_frequency[pt] = node_frequency.get(pt, 0) + 1

    route_stops = []
    for route in routes:
        if not route:
            route_stops.append([])
            continue

        mandatory_idx = [0]
        for i, pt in enumerate(route[1:-1], start=1):
            if node_frequency.get(pt, 0) > 1:
                mandatory_idx.append(i)
        mandatory_idx.append(len(route) - 1)
        mandatory_idx = sorted(set(mandatory_idx))

        stops = []
        for i in range(len(mandatory_idx) - 1):
            start_idx = mandatory_idx[i]
            end_idx = mandatory_idx[i + 1]
            start_pt = route[start_idx]
            end_pt = route[end_idx]
            stops.append(start_pt)
            gap = euclidean_distance(start_pt, end_pt)
            if gap > extra_stop_gap:
                n_extra = math.ceil(gap / extra_stop_gap) - 1
                for j in range(1, n_extra + 1):
                    frac = j / (n_extra + 1)
                    new_stop = (start_pt[0] + frac * (end_pt[0] - start_pt[0]),
                                start_pt[1] + frac * (end_pt[1] - start_pt[1]))
                    stops.append(new_stop)
        stops.append(route[mandatory_idx[-1]])
        route_stops.append(stops)

    global_stops = []
    route_stop_indices = []
    for stops in route_stops:
        current_indices = []
        for pt in stops:
            found_idx = None
            for idx, gst in enumerate(global_stops):
                if euclidean_distance(pt, gst) < merge_tolerance:
                    found_idx = idx
                    new_x = (gst[0] + pt[0]) / 2
                    new_y = (gst[1] + pt[1]) / 2
                    global_stops[idx] = (new_x, new_y)
                    break
            if found_idx is None:
                global_stops.append(pt)
                found_idx = len(global_stops) - 1
            current_indices.append(found_idx)
        route_stop_indices.append(current_indices)

    return global_stops, route_stop_indices


def to_coord(pt):
    return (pt.x, pt.y) if hasattr(pt, 'x') else pt


def plot_final_routes(final_bus_routes, augmented_network):
    routes_coords = []
    for route in final_bus_routes:
        routes_coords.append([to_coord(pt) for pt in route])

    global_stops, route_stop_indices = generate_global_stop_candidates(routes_coords,
                                                                       extra_stop_gap=500,
                                                                       merge_tolerance=20)

    stop_routes = {}
    for route_idx, stop_idxs in enumerate(route_stop_indices):
        for stop_idx in stop_idxs:
            stop_routes.setdefault(stop_idx, set()).add(route_idx + 1)

    stops_all_x = [pt[0] for pt in global_stops]
    stops_all_y = [pt[1] for pt in global_stops]
    stops_all_text = [
        f"Stop {idx + 1}<br>Routes: {', '.join(map(str, sorted(stop_routes.get(idx, []))))}"
        for idx in range(len(global_stops))
    ]

    stops_by_route = []
    for route_num in range(1, len(final_bus_routes) + 1):
        xs = []
        ys = []
        texts = []
        for idx, pt in enumerate(global_stops):
            if route_num in stop_routes.get(idx, []):
                xs.append(pt[0])
                ys.append(pt[1])
                texts.append(f"Stop {idx + 1}<br>Routes: {', '.join(map(str, sorted(stop_routes.get(idx, []))))}")
        stops_by_route.append((xs, ys, texts))

    edge_traces = []
    for u, v, data in augmented_network.edges(data=True):
        geom = data.get('geometry')
        if not isinstance(geom, LineString):
            geom = LineString([Point(u), Point(v)])
        x, y = geom.xy
        edge_traces.append(
            go.Scatter(
                x=list(x),
                y=list(y),
                mode='lines',
                line=dict(color='gray', width=1),
                hoverinfo='none'
            )
        )

    cmap = cm.get_cmap('tab20', len(final_bus_routes))
    route_traces = []
    for idx, route in enumerate(final_bus_routes):
        pts = [Point(n) if not isinstance(n, Point) else n for n in route]
        line = LineString(pts)
        x, y = line.xy
        route_traces.append(
            go.Scatter(
                x=list(x),
                y=list(y),
                mode='lines',
                line=dict(color=rgba_to_rgb_str(cmap(idx)), width=3),
                name=f"Route {idx + 1}",
                visible=False
            )
        )

    stops_all_trace = go.Scatter(
        x=stops_all_x,
        y=stops_all_y,
        mode='markers',
        marker=dict(size=8, color='black'),
        name="Stops (All Routes)",
        text=stops_all_text,
        hoverinfo='text',
        visible=False
    )

    stops_traces = []
    for route_idx, (xs, ys, texts) in enumerate(stops_by_route):
        stops_traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode='markers',
                marker=dict(size=10, color='black'),
                name=f"Stops (Route {route_idx + 1})",
                text=texts,
                hoverinfo='text',
                visible=False
            )
        )

    fig = go.Figure(
        data=edge_traces + route_traces + [stops_all_trace] + stops_traces
    )

    n_edges = len(edge_traces)
    n_routes = len(route_traces)
    n_stops_total = 1 + len(stops_traces)
    buttons = []

    visible_all = (
            [True] * n_edges +
            [True] * n_routes +
            ([True] + [False] * (n_stops_total - 1))
    )
    buttons.append(dict(
        label="All Routes",
        method="update",
        args=[{"visible": visible_all},
              {"title": "Final Bus Routes - All Routes"}]
    ))

    for i in range(len(final_bus_routes)):
        route_vis = [False] * n_routes
        route_vis[i] = True
        stops_vis = [False] * n_stops_total
        stops_vis[i + 1] = True
        visible = (
                [True] * n_edges +
                route_vis +
                stops_vis
        )
        buttons.append(dict(
            label=f"Route {i + 1}",
            method="update",
            args=[{"visible": visible},
                  {"title": f"Final Bus Routes - Route {i + 1}"}]
        ))

    fig.update_layout(
        title="Final Bus Routes",
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=1.1,
            y=1
        )],
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        template="plotly_white"
    )
    fig.write_html("bus_routes_with_stops.html")
    fig.show()


def main():
    roads, average_length = verify_data()
    mst, hub_scores, G = make_graph(roads, average_length)
    aug_network = optimize_transit(mst, hub_scores, G, roads)
    final_bus_routes, cmap, edge_traces = display_data(aug_network)
    display_bus_routes(final_bus_routes, cmap, edge_traces)
    plot_final_routes(final_bus_routes, aug_network)

if ___name___ == '___main___':
    main()
