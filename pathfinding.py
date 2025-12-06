import osmnx as ox
import networkx as nx
import time
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt


def calculate_haversine_distance(u, v, G_data):
    """
    Calculates the Haversine distance between two nodes (u and v) in the graph G_data.
    This serves as the admissible heuristic h(n) for the A* algorithm.
    """
    # Get coordinates from node attributes
    lon1, lat1 = G_data.nodes[u]['x'], G_data.nodes[u]['y']
    lon2, lat2 = G_data.nodes[v]['x'], G_data.nodes[v]['y']
    
    R = 6371000  # Radius of Earth in meters

    lon1_rad, lat1_rad = radians(lon1), radians(lat1)
    lon2_rad, lat2_rad = radians(lon2), radians(lat2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    return distance


place = "Pike Place Market, Seattle, Washington, USA"
print(f"Loading street network graph for: {place}...")

center_lat = 47.6075
center_lon = -122.3375
distance_radius = 600 

G = ox.graph_from_point(
    (center_lat, center_lon), 
    dist=distance_radius, 
    network_type="drive"
)
print("Graph loaded successfully.")

print("\nGenerating full graph visualization...")
fig_full, ax_full = ox.plot_graph(
    G,
    node_size=0,
    edge_linewidth=0.5,
    bgcolor='w',
    edge_color='gray',
    show=False,
    close=False,
    save=True,
    filepath='full_road_network_zoomed.png',
    dpi=300
)
# Force display of the full map
plt.show() 
plt.close(fig_full) 
print("Full road network visualization saved to 'full_road_network_zoomed.png' and is viewable.")


# Start and End Points 
start_lat, start_lon = 47.6091, -122.3402
end_lat, end_lon = 47.6060, -122.3330

orig_node = ox.nearest_nodes(G, start_lon, start_lat)
dest_node = ox.nearest_nodes(G, end_lon, end_lat)

print(f"\nScenario defined:")
print(f"Start Node ID: {orig_node} (A)")
print(f"End Node ID: {dest_node} (B)")

#A* qnd search time
num_runs = 10
total_time = 0
shortest_path = None

print(f"\nRunning A* search {num_runs} times to average computation time...")

# Heuristic
haversine_heuristic = lambda u, v: calculate_haversine_distance(u, v, G)

for i in range(num_runs):
    start_time = time.perf_counter()
    
    try:
        path = nx.astar_path(
            G, 
            source=orig_node, 
            target=dest_node, 
            weight="length",
            heuristic=haversine_heuristic
        )
    except nx.NetworkXNoPath:
        print("Error: No path found between the nodes.")
        break
    
    end_time = time.perf_counter()
    total_time += (end_time - start_time)
    shortest_path = path

# Distance
if shortest_path:
    total_distance = sum(G.edges[u, v, 0]['length'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))
    average_time = total_time / num_runs

    print("\n--- A* Search Results ---")
    print(f"Optimal Path Nodes (first 5): {shortest_path[:5]}...")
    print(f"Optimal Path Nodes (last 5): ...{shortest_path[-5:]}")
    print(f"Total Distance/Cost: {total_distance:,.2f} meters")
    print(f"Average Computation Time: {average_time * 1000:,.4f} milliseconds (over {num_runs} runs)")

    print("Generating A* route visualization...")
    
    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_linewidth=0.5,
        bgcolor='w',
        edge_color='gray',
        show=False,
        close=False,
        save=False
    )
    
    # Manually plot the route edges using Matplotlib
    for u, v in zip(shortest_path[:-1], shortest_path[1:]):
        edge_data = G.get_edge_data(u, v)
        coords = edge_data[0].get('geometry', None) 
        
        if coords:
            x, y = coords.xy
            ax.plot(x, y, color='r', linewidth=4, alpha=0.7, zorder=5)
        else:
            x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
            x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
            ax.plot([x1, x2], [y1, y2], color='r', linewidth=4, alpha=0.7, zorder=5)


    # Manually plot the origin/destination nodes and labels using Matplotlib
    # Origin Node (A)
    ax.scatter(
        G.nodes[orig_node]['x'], 
        G.nodes[orig_node]['y'], 
        s=500, 
        color='green', 
        marker='o', 
        zorder=6
    )
    ax.text(
        G.nodes[orig_node]['x'], 
        G.nodes[orig_node]['y'], 
        'A (Start)', 
        color='black', 
        fontsize=12, 
        ha='left', 
        va='center', 
        weight='bold', 
        zorder=7
    )
    
    # Destination Node (B)
    ax.scatter(
        G.nodes[dest_node]['x'], 
        G.nodes[dest_node]['y'], 
        s=500, 
        color='blue', 
        marker='o', 
        zorder=6
    )
    ax.text(
        G.nodes[dest_node]['x'], 
        G.nodes[dest_node]['y'], 
        'B (End)', 
        color='black', 
        fontsize=12, 
        ha='right', 
        va='center', 
        weight='bold', 
        zorder=7
    )
    
    fig.savefig('a_star_path_labeled.png', dpi=300, bbox_inches='tight')
    plt.show() 
    plt.close(fig) 
    print("A* route visualization is now viewable.")