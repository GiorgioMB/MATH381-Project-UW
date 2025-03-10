import math
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.colors import Normalize
from shapely.geometry import Point, LineString
from typing import Tuple, List, Dict, Any, Union

def verify_data() -> Tuple[gpd.GeoDataFrame, float]:
    """
    Description:
    Reads the dataset and ensures it uses a projected coordinate system, checks for the presence of the 'ADT' (Average Daily Traffic) column,
    computes busyness values and normalizes them for comparison, and calculates road lengths, computing the average road length for reference.
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



def extract_lines(geom: LineString) -> List[LineString]:
    """
    Input:
    - geom: a LineString or MultiLineString geometry object.

    Output:
    - Individual LineString components from the input geometry.

    Description:
    Extract and yield individual LineStrings from a given geometry.
    """
    # Check if the geometry is a single LineString
    if geom.geom_type == 'LineString':
        yield geom

    # If it's a MultiLineString, iterate through its components
    elif geom.geom_type == 'MultiLineString':
        for line in geom.geoms:
            yield line



def vertex_key(coord: Tuple[float, float], precision=3) -> Tuple[float, float]:
    """
    Generate a unique key for a vertex by rounding its coordinates to remove floating-point precision errors.
    """
    return round(coord[0], precision), round(coord[1], precision)



def compute_edge_weight(edge_data: Dict[str, Any], start_hub: float, 
                        end_hub: float, avg_length: float,
                        lambda_m: float =1.0, lambda_c: float = 1.0, beta: float = 2.0,
                        base_freq: int = 5, scale: int = 10, gamma: float = 0.2) -> float:
    """
    Input:
    - edge_data: a dictionary containing edge attributes (length, busyness_norm).
    - start_hub: the hub score of the starting node.
    - end_hub: the hub score of the ending node.
    - avg_length: the average road length in the dataset.
    - lambda_m: the mismatch penalty weight.
    - lambda_c: the convenience penalty weight.
    - beta: the hub bonus weight.
    - base_freq: the base frequency of transit service.
    - scale: the scaling factor for frequency estimation.
    - gamma: the fare-related cost weight.

    Output:
    - The computed edge weight based on multiple factors.

    Description:
    Compute the weight of an edge based on its length, busyness, hub scores of endpoints, penalising
    mismatched busyness, applying a convenience penalty for busy hubs, providing a bonus for connecting
    high-busyness hubs, estimating transit frequency, and calculating a fare-related cost.
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



def make_graph(roads: gpd.GeoDataFrame, avg_length:float) -> Tuple[nx.Graph, Dict[Any, float], nx.Graph]:
    """
    Input:
    - roads: a GeoDataFrame containing road network data.
    - avg_length: the average road length in the dataset.

    Output:
    - mst: the minimum spanning tree of the road network graph.
    - hub_scores: a dictionary of hub scores for each node in the graph.
    - G: the full road network graph.

    Description:
    Constructs a graph from the road network data, computes hub scores for each node based on incident edge busyness,
    normalizes hub scores, assigns hub scores as node attributes, computes edge weights based on multiple factors,
    and calculates the minimum spanning tree of the graph.
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



def assign_frequency(edge_data: Dict[str, Any], start_hub: float, 
                     end_hub: float, base_freq: int = 5, scale: int = 10) -> float:
    """
    Input:
    - edge_data: a dictionary containing edge attributes (busyness_norm).
    - start_hub: the hub score of the starting node.
    - end_hub: the hub score of the ending node.
    - base_freq: the base frequency of transit service.
    - scale: the scaling factor for frequency estimation.
    
    Output:
    - The computed frequency of transit service on the edge.

    Description:
    Computes the frequency of transit service on an edge based on busyness, hub scores of endpoints, base frequency,
    and a scaling factor.                                                                                                                                                                                                                                                   
    """
    # Compute an average factor based on road busyness and hub scores
    avg_factor = (edge_data['busyness_norm'] + start_hub + end_hub) / 3.0

    # Calculate the frequency based on the base frequency and scaled busyness
    freq = base_freq + scale * avg_factor

    return freq



def get_endpoint_coords(route: List[Any], G: nx.Graph) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Input:
    - route: a list of nodes (as coordinate tuples or nodes with 'x','y').
    - G: a networkx graph with node attributes.

    Output:
    - The coordinates of the start and end points of the route.

    Description:
    Retrieve the coordinates of the start and end points of a route from the graph's node attributes,
    handling both coordinate tuples and nodes with 'x' and 'y' attributes.
    """
    
    def get_coord(node: Tuple[float, float]) -> Tuple[float, float]:
        """
        Helper function to retrieve coordinates from a node.
        """
        # If the node is already a coordinate tuple, return it
        if isinstance(node, tuple) and len(node) == 2:
            return node

        # Otherwise, retrieve coordinates from the graph's node attributes
        node_data = G.nodes[node]
        return node_data.get('x', 0), node_data.get('y', 0)

    return get_coord(route[0]), get_coord(route[-1])



def euclidean_distance(p1: Tuple[float,float], p2: Tuple[float,float]) -> float:
    """Helper function to compute a 2-dimensional Euclidean distance."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])



def compute_geometric_penalty(route: List[Tuple], close_threshold:float=10) -> float:
    """
    Inputs:
    - route: a list of nodes (as coordinate tuples).
    - close_threshold: the threshold distance for considering a route 'closed'.

    Output:
    - The computed geometric penalty for the route.

    Description:
    Computes a geometric penalty for a route based on the distance between its start and end points,
    and the average turning angles at internal vertices, penalizing routes that are not closed or have erratic turning angles.
    """
    # Convert route points into NumPy arrays for vector calculations
    pts = [np.array(p) for p in route]

    # Compute the openness penalty based on the distance between start and end points
    openness_penalty = 0.0
    total_dist = 0
    for i in range(len(route)-1):
        start = route[i]
        end = route[i+1]
        total_dist += euclidean_distance(start, end)
    if total_dist > close_threshold:
        openness_penalty = total_dist - close_threshold

    # Compute the turning penalty based on the average turning angle at internal vertices
    total_turn = 0.0
    count = 0
    # Iterate through internal points to compute turning angles
    for i in range(1, len(pts) - 1):
        # Vector from previous point to current point
        v1 = pts[i] - pts[i - 1]
        # Vector from current point to next point
        v2 = pts[i + 1] - pts[i]
        # Compute the product of vector norms
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        # Avoid division by zero if one of the vectors has zero length
        if norm_product == 0:
            continue
        # Compute the turning angle using the dot product formula and clip to avoid errors
        angle = np.arccos(np.clip(np.dot(v1, v2) / norm_product, -1, 1))
        # Accumulate absolute turning angles
        total_turn += abs(angle)
        count += 1

    # Compute the average turning penalty, avoiding division by zero
    turning_penalty = (total_turn / count) if count > 0 else 0.0

    return openness_penalty + turning_penalty



def merge_two_routes_improved(route1: List[Tuple[float,float]], route2: List[Tuple[float,float]], 
                              G: nx.Graph, forbidden_penalty: float = 100.0,
                              merge_length_factor: int = 30, geometric_factor:float = 1.0) -> Tuple[List[Tuple[float,float]], float]:
    """
    Inputs:
    - route1: a list of nodes (as coordinate tuples).
    - route2: a list of nodes (as coordinate tuples).
    - G: a networkx graph with edge attributes.
    - forbidden_penalty: the penalty for using forbidden edges.
    - merge_length_factor: the factor for merge length in the total cost.
    - geometric_factor: the factor for geometric deviation in the total cost.

    Output:
    - The merged route with the lowest total cost.
    - The total cost of the merged route.

    Description:
    Attempts to merge two routes by finding the shortest path between their endpoints, considering
    forbidden edges and applying penalties for merge length and geometric deviation.
    """
    # Initialize best merge variables
    best_total_cost = math.inf
    best_merge = None

    # Define custom weight function for pathfinding
    def custom_weight(u, v, d):
        # Base weight is the edge lengt
        base = d.get('length', 0)
        # Apply penalty for forbidden edges
        if d.get('forbidden', False):
            base += forbidden_penalty
        return base

    # Compute total Euclidean distance of a given route
    def compute_route_length(route):
        total = 0.0
        for i in range(len(route) - 1):
            total += euclidean_distance(route[i], route[i + 1])
        return total

     # Generate possible merging orientations
    options = [
        (route1, route2, route1[-1], route2[0]),
        (route1, list(reversed(route2)), route1[-1], list(reversed(route2))[0]),
        (list(reversed(route1)), route2, list(reversed(route1))[-1], route2[0]),
        (list(reversed(route1)), list(reversed(route2)), list(reversed(route1))[-1], list(reversed(route2))[0])
    ]

    # Iterate through merging options
    for r1, r2, u, v in options:
        try:
            # Find shortest path between endpoints using the custom weight function
            path = nx.shortest_path(G, source=u, target=v, weight=custom_weight)
            spath_cost = nx.shortest_path_length(G, source=u, target=v, weight=custom_weight)
        # Skip if no valid path exists
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        
        # Create the merged route and compute route length and geometric penalty
        merged_route = r1 + path[1:] + r2
        route_length = compute_route_length(merged_route)
        geom_pen = compute_geometric_penalty(merged_route)

        # Compute total cost considering shortest path, length, and geometric deviation
        total_cost = spath_cost + merge_length_factor * route_length + geometric_factor * geom_pen

        # Update the best merge if a lower cost is found
        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_merge = merged_route

    return best_merge, best_total_cost



def optimize_transit(mst: nx.Graph, hub_scores: Dict[Any, float],
                     G: nx.Graph, roads: gpd.GeoDataFrame) -> nx.Graph:
    """
    Input:
    - mst: the minimum spanning tree of the road network graph.
    - hub_scores: a dictionary of hub scores for each node in the graph.
    - G: the full road network graph.
    - roads: a GeoDataFrame containing road network data.

    Output:
    - The optimized transit network with augmented edges.

    Description:
    Assigns service frequency and average wait time to MST edges based on hub scores, identifies redundant edges
    that can be added back to improve connectivity, creates an augmented network by adding selected redundant edges
    to the MST, and prepares data for visualization, plotting the optimized transit network with average wait times.
    """
    # Assign service frequency and average wait time to MST edges
    for u, v, data in mst.edges(data=True):
        start_hub = hub_scores.get(u, 0)
        end_hub = hub_scores.get(v, 0)
        freq = assign_frequency(data, start_hub, end_hub)
        data['service_frequency'] = freq
        data['avg_wait_time'] = 60.0 / freq

    # Identify redundant edges that can be added back to improve connectivity
    redundancy_threshold = 0.9
    redundant_edges = []
    for u, v, data in G.edges(data=True):
        # Ignore edges already in the MST
        if mst.has_edge(u, v):
            continue
        if hub_scores.get(u, 0) > redundancy_threshold and hub_scores.get(v, 0) > redundancy_threshold:
            redundant_edges.append((u, v, data))

    # Create an augmented network by adding selected redundant edges to the MST
    augmented_network = mst.copy()
    for u, v, data in redundant_edges:
        start_hub = hub_scores.get(u, 0)
        end_hub = hub_scores.get(v, 0)
        freq = assign_frequency(data, start_hub, end_hub)
        data['service_frequency'] = freq
        data['avg_wait_time'] = 60.0 / freq
        augmented_network.add_edge(u, v, **data)

     # Prepare data for visualization
    aug_edges = []
    wait_times = []
    for u, v, data in augmented_network.edges(data=True):
        geom = data.get('geometry')
        if not isinstance(geom, LineString):
            geom = LineString([Point(u), Point(v)])
        aug_edges.append(geom)
        wait_times.append(data.get('avg_wait_time', 60))

     # Convert edges into a GeoDataFrame
    aug_gdf = gpd.GeoDataFrame(geometry=aug_edges, crs=roads.crs)

    # Normalize wait times for color mapping
    norm = Normalize(vmin=min(wait_times), vmax=max(wait_times))
    cmap = cm.get_cmap('coolwarm')

    # Plot the optimized transit network
    fig, ax = plt.subplots(figsize=(12, 12))
    roads.plot(ax=ax, color="black", linewidth=1, label="Original Road Network")

    # Color edges based on wait times
    for geom, wt in zip(aug_edges, wait_times):
        ax.plot(*geom.xy, color=cmap(norm(wt)), linewidth=1.2)

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Average Wait Time (min)")

    ax.set_title("MST Edges with Average Wait Time")
    ax.legend()
    plt.savefig("heatmap_network.png")
    plt.show()
    return augmented_network



def reduce_route_count(routes:List[List[Tuple[float, float]]], G: nx.Graph, target_count=300, dist_threshold=110, max_iterations=2000,
                       forbidden_penalty=100.0, merge_length_factor=30, geometric_factor=1.0) -> List[List[Tuple[float, float]]]:
    """
    Input:
    - routes: a list of bus routes (each a list of node coordinates).
    - G: a networkx graph with edge attributes.
    - target_count: the desired number of routes to reduce to.
    - dist_threshold: the distance threshold for merging routes.
    - max_iterations: the maximum number of iterations to attempt merging.
    - forbidden_penalty: the penalty for using forbidden edges.
    - merge_length_factor: the factor for merge length in the total cost.
    - geometric_factor: the factor for geometric deviation in the total cost.

    Output:
    - The reduced list of bus routes.

    Description:
    Iteratively merges routes that are close together based on a distance threshold, attempting to reduce the
    total number of routes to a target count while applying penalties for forbidden edges, merge length, and geometric deviation.
    """
    # Create a copy of the routes list to avoid modifying the original data
    routes = routes.copy()
    iteration = 0

    # Continue merging until the target count is reached or max iterations are exceeded
    while len(routes) > target_count and iteration < max_iterations:
        best_total_cost = math.inf
        best_pair = None
        best_merged = None

        # Precompute endpoint coordinates for all routes
        endpoints = [get_endpoint_coords(route, G) for route in routes]

        # Iterate through all pairs of routes to find the best merge
        for i in range(len(routes)):
            start_i, end_i = endpoints[i]
            for j in range(i + 1, len(routes)):
                start_j, end_j = endpoints[j]

                # Check if the routes are close enough to be merged
                if (euclidean_distance(end_i, start_j) < dist_threshold or
                        euclidean_distance(end_i, end_j) < dist_threshold or
                        euclidean_distance(start_i, start_j) < dist_threshold or
                        euclidean_distance(start_i, end_j) < dist_threshold):

                    # Attempt to merge the two routes and compute the cost
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

        # If no valid merge was found, terminate the loop
        if best_pair is None:
            break
        
        # Extract the best pair of routes to merge
        i, j = best_pair

        # Create a new list of routes excluding the merged ones and adding the new merged route
        new_routes = [routes[k] for k in range(len(routes)) if k not in best_pair]
        new_routes.append(best_merged)
        routes = new_routes

        # Increment iteration count and print progress
        iteration += 1
        print(f"Iteration {iteration:3d}: Merged routes {i} and {j}; Total routes now = {len(routes)}", end='\r')
    return routes



def prepare_edge_demand(G: nx.Graph) -> Dict[Tuple, int]:
    """
    Input:
    - G: a networkx graph with edge attributes.

    Output:
    - A dictionary of edge demand values.

    Description:
    Computes the demand for each edge in the graph based on service frequency, ensuring a minimum demand of 1.
    """
    # Initialize an empty dictionary to store edge demand values
    edge_demand = {}

    # Iterate over all edges in the graph
    for u, v, data in G.edges(data=True):
        # Retrieve the service frequency, ensuring a minimum demand of 1
        demand = max(1, int(round(data.get('service_frequency', 1))))
        # Store edges in a sorted tuple format to avoid duplication (u, v) == (v, u)
        key = tuple(sorted((u, v)))
        # Accumulate demand for each unique edge
        edge_demand[key] = edge_demand.get(key, 0) + demand
    return edge_demand



def generate_bus_routes(G: nx.Graph) -> List[List[Any]]:
    """
    Input:
    - G: a networkx graph with edge attributes.

    Output:
    - A list of generated bus routes.

    Description:
    Generates bus routes by iteratively selecting the edge with the highest remaining demand,
    extending the route from both ends to find the highest-demand neighboring edges, and adding
    the finalized route to the list of bus routes.
    """
    # Compute initial edge demand from the graph
    edge_demand = prepare_edge_demand(G)
    routes = []

    # Continue generating routes while there are edges with demand remaining
    while any(d > 0 for d in edge_demand.values()):
        # Select the edge with the highest remaining demand
        candidate_edge, demand_val = max(edge_demand.items(), key=lambda item: item[1])
        if demand_val <= 0:
            break
        
        # Extract the two endpoints of the selected edge and decrement its demand
        u, v = candidate_edge
        edge_demand[candidate_edge] -= 1

        # Initialize a new bus route with the selected edge
        route = [u, v]

        # Function to extend the route from a given endpoint
        def extend_route(endpoint, front=True):
            current = endpoint
            while True:
                best_n = None
                best_demand = 0
                # Check all neighboring nodes to find the edge with the highest demand
                for neighbor in G.neighbors(current):
                    key = tuple(sorted((current, neighbor)))
                    if edge_demand.get(key, 0) > best_demand:
                        best_demand = edge_demand[key]
                        best_n = neighbor
                # Stop extending if no valid high-demand edge is found
                if best_n is None or best_demand <= 0:
                    break
                # Extend route at the front or back based on the parameter
                if front:
                    route.insert(0, best_n)
                else:
                    route.append(best_n)
                # Reduce demand for the selected edge
                key = tuple(sorted((current, best_n)))
                edge_demand[key] -= 1
                current = best_n

        # Extend the route from both ends and add the finalized route to the list of bus routes
        extend_route(route[0], front=True)
        extend_route(route[-1], front=False)
        routes.append(route)
    return routes



def display_data(augmented_network : nx.Graph) -> Tuple[List[List[Tuple[float, float]]], cm, List[go.Scatter]]:
    """
    Input:
    - augmented_network: a networkx graph representing the augmented road network.

    Output:
    - final_bus_routes: a list of final bus routes.
    - cmap: a colormap for visualizing bus routes.
    - edge_traces: a list of plotly traces for road edges.

    Description:
    Displays the final bus routes on a plot and in an interactive plotly figure.
    """

    initial_bus_routes = generate_bus_routes(augmented_network)
    print("Initial number of routes:", len(initial_bus_routes))

    final_bus_routes = reduce_route_count(initial_bus_routes, augmented_network, target_count=300)
    final_bus_routes = reduce_route_count(final_bus_routes, augmented_network, target_count=300, dist_threshold=500)
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



def rgba_to_rgb_str(rgba: Tuple[float, float, float, float]) -> str:
    """
    Input:
    - rgba: an RGBA color tuple.

    Output:
    - An RGB string for Plotly.

    Description:
    Convert an RGBA color tuple to an RGB string for Plotly.
    """

    r, g, b, _ = rgba
    return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"



def display_bus_routes(final_bus_routes: List[List[Tuple[float, float]]], cmap: cm, edge_traces: List[go.Scatter]):
    """
    Input:
    - final_bus_routes: a list of final bus routes.
    - cmap: a colormap for visualizing bus routes.
    - edge_traces: a list of plotly traces for road edges.

    Output:
    - A Plotly figure displaying the final bus routes.

    Description:
    Displays the final bus routes in an interactive Plotly figure with buttons to toggle visibility.
    """

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



def generate_global_stop_candidates(routes: List[Tuple[float,float]], extra_stop_gap=500, merge_tolerance=20) -> Tuple[List[Tuple[float,float]], List[List[int]]]:
    """
    Input
    - routes: a list of bus routes (each a list of node coordinates).
    - extra_stop_gap: the maximum gap between stops before inserting additional stops.
    - merge_tolerance: the maximum distance between stops to merge them.

    Output
    - global_stops: a list of global stop candidates.
    - route_stop_indices: a list of indices mapping stops to routes.

    Description
    Generates global stop candidates by identifying mandatory stops at route start, end, and intersections,
    inserting additional stops if gaps between mandatory stops are too large, and merging stops that are within
    the merge tolerance distance.
    """
    # Count how many routes pass through each node
    node_frequency = {}
    for route in routes:
        for pt in route:
            node_frequency[pt] = node_frequency.get(pt, 0) + 1

    route_stops = []

    # Determine stop locations along each route
    for route in routes:
        if not route:
            route_stops.append([])
            continue

        # Identify mandatory stops at route start, end, and intersections
        mandatory_idx = [0]
        for i, pt in enumerate(route[1:-1], start=1):
            if node_frequency.get(pt, 0) > 1:
                mandatory_idx.append(i)
        mandatory_idx.append(len(route) - 1)
        mandatory_idx = sorted(set(mandatory_idx))

        stops = []

        # Insert additional stops if gaps between mandatory stops are too large
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
        
        # Ensure the last mandatory stop is included
        stops.append(route[mandatory_idx[-1]])
        route_stops.append(stops)

    global_stops = []
    route_stop_indices = []

    # Merge stops that are within the merge_tolerance
    for stops in route_stops:
        current_indices = []
        for pt in stops:
            found_idx = None
            for idx, gst in enumerate(global_stops):
                if euclidean_distance(pt, gst) < merge_tolerance:
                    found_idx = idx
                    # Average the stop positions to merge them
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


def to_coord(pt: Union[Tuple[float, float], Point]) -> Tuple[float, float]:
    """
    Input:
    - pt: a point as a tuple or a Shapely Point.

    Output:
    - A coordinate

    Description:
    Convert a point to a coordinate tuple.
    """
    return (pt.x, pt.y) if hasattr(pt, 'x') else pt


def plot_final_routes(final_bus_routes: List[List[Tuple[float, float]]], augmented_network: nx.Graph) -> None:
    """
    Input:
    - final_bus_routes: a list of final bus routes.
    - augmented_network: a networkx graph representing the augmented road network.

    Output:
    - A Plotly figure displaying the final bus routes with stops.

    Description:
    Plots the final bus routes with stop locations on an interactive Plotly figure
    with buttons to toggle visibility between routes and stops.
    """
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
    """
    Main function to run the entire pipeline.   
    """
    roads, average_length = verify_data()
    mst, hub_scores, G = make_graph(roads, average_length)
    aug_network = optimize_transit(mst, hub_scores, G, roads)
    final_bus_routes, cmap, edge_traces = display_data(aug_network)
    display_bus_routes(final_bus_routes, cmap, edge_traces)
    plot_final_routes(final_bus_routes, aug_network)

if ___name___ == '___main___':
    main()
