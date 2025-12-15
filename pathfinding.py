import osmnx as ox
import networkx as nx
from math import radians, sin, cos, sqrt, atan2

class PathFinder:
    _shared_graph = None 

    def __init__(self, center_lat=47.6062, center_lon=-122.3321, dist=5000):
        """
        Initializes the road network graph.
        :param dist: Radius in meters. Increased to 5000m to cover app.py mock data.
        """
        if PathFinder._shared_graph is None:
            print(f"Loading street network graph (Radius: {dist}m)...")

            G = ox.graph_from_point(
                (center_lat, center_lon), 
                dist=dist, 
                network_type="drive"
            )
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
            PathFinder._shared_graph = G
            print("Graph loaded successfully.")
        
        self.G = PathFinder._shared_graph

    def _haversine_heuristic(self, u, v):
        """
        Calculates Haversine distance between two nodes for A* heuristic.
        """
        lon1, lat1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
        lon2, lat2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
        
        R = 6371000  # Earth radius in meters
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def get_astar_distance(self, start_coords, end_coords):
        """
        Finds the A* path distance between two (lat, lon) tuples.
        """
        orig_node = ox.nearest_nodes(self.G, start_coords[1], start_coords[0])
        dest_node = ox.nearest_nodes(self.G, end_coords[1], end_coords[0])

        if orig_node == dest_node:
            return 0.0

        try:
            length = nx.astar_path_length(
                self.G, 
                source=orig_node, 
                target=dest_node, 
                weight="length",
                heuristic=self._haversine_heuristic
            )
            return length
        except nx.NetworkXNoPath:
            return self._haversine_heuristic(orig_node, dest_node)

if __name__ == "__main__":
    pf = PathFinder()
    d = pf.get_astar_distance((47.6091, -122.3402), (47.6060, -122.3330))
    print(f"A* Distance: {d:.2f} meters")