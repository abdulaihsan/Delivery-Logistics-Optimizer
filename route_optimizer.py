import random
import numpy as np
import networkx as nx
import osmnx as ox
import pandas as pd

from search_model import calculate_haversine_distance
from model_interpretability import rf_model, encoder

class RouteOptimizerGA:
    def __init__(self, locations, population_size=50, mutation_rate=0.01, generations=100):
        """
        Genetic Algorithm for VRPTW.
        :param locations: List of tuples [(lat, lon), ...] or Node IDs
        """
        self.locations = locations
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = []
        
    def create_individual(self):
        """Creates a random permutation of locations (excluding depot if needed)."""
        indices = list(range(1, len(self.locations)))
        random.shuffle(indices)
        return [0] + indices + [0]

    def initialize_population(self):
        self.population = [self.create_individual() for _ in range(self.pop_size)]

    def get_trip_cost(self, u_idx, v_idx):
        """
        INTEGRATION POINT: 
        Uses A* for distance and ML Model for time.
        """
        u_coords = self.locations[u_idx]
        v_coords = self.locations[v_idx]
        
        from math import radians, sin, cos, sqrt, atan2
        R = 6371000
        lat1, lon1 = radians(u_coords[0]), radians(u_coords[1])
        lat2, lon2 = radians(v_coords[0]), radians(v_coords[1])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        
        # 2. Time from ML Logic
        miles = distance / 1609.34
        current_hour = 9   
        day_of_week = 0    # 0 = Monday

        raw_input = pd.DataFrame({
            'MILES': [miles],
            'START_HOUR': [current_hour],
            'DAY_OF_WEEK': [day_of_week],
            'CATEGORY': ['Business'],   # Default value
            'PURPOSE': ['Unknown'],     # Default value ('Unknown' was used in training cleaning)
            'START_LOC': ['Other'],     # Default value (matches "Other" bucket from training)
            'STOP_LOC': ['Other']       # Default value
        })

        categorical_cols = ['CATEGORY', 'PURPOSE', 'START_LOC', 'STOP_LOC']
        encoded_cats = encoder.transform(raw_input[categorical_cols])
        
        encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out())
        
        X_numeric = raw_input[['MILES', 'START_HOUR', 'DAY_OF_WEEK']]
        X_input = pd.concat([X_numeric, encoded_df], axis=1)

        predicted_time = rf_model.predict(X_input)[0]
        
        return distance, predicted_time

    def fitness(self, individual):
        """Calculates total cost (Distance + Time penalties)."""
        total_distance = 0
        total_time = 0
        
        for i in range(len(individual) - 1):
            u, v = individual[i], individual[i+1]
            d, t = self.get_trip_cost(u, v)
            total_distance += d
            total_time += t
            
        return 1 / (total_distance + 1e-5)

    def selection(self):
        """Tournament Selection"""
        k = 3
        selected = []
        for _ in range(self.pop_size):
            aspirants = random.choices(self.population, k=k)
            best = max(aspirants, key=self.fitness)
            selected.append(best)
        return selected

    def crossover(self, parent1, parent2):
        """Ordered Crossover (OX1) for TSP/VRP"""
        size = len(parent1) - 2 
        start, end = sorted(random.sample(range(1, size + 1), 2))
        
        child = [None] * len(parent1)
        child[0], child[-1] = 0, 0
        child[start:end] = parent1[start:end]
        
        p2_genes = [item for item in parent2[1:-1] if item not in child]
        pointer = 0
        for i in range(1, len(child)-1):
            if child[i] is None:
                child[i] = p2_genes[pointer]
                pointer += 1
        return child

    def mutate(self, individual):
        """Swap Mutation"""
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(1, len(individual)-1), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

    def run(self):
        print(f"Initializing GA optimization for {len(self.locations)} locations...")
        self.initialize_population()
        
        for gen in range(self.generations):
            parents = self.selection()
            
            next_pop = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = parents[i], parents[i+1]
                child1 = self.crossover(p1, p2)
                child2 = self.crossover(p2, p1)
                next_pop.extend([child1, child2])
            
            for ind in next_pop:
                self.mutate(ind)
                
            self.population = next_pop
            
            if gen % 20 == 0:
                best_fit = max(self.population, key=self.fitness)
                print(f"Generation {gen}: Best Route = {best_fit}")

        best_route = max(self.population, key=self.fitness)
        print(f"Optimization Complete. Optimal Route Sequence: {best_route}")
        return best_route

if __name__ == "__main__":
    # Mock Data 
    locations = [
        (47.6075, -122.3375), 
        (47.6091, -122.3402),
        (47.6060, -122.3330),
        (47.6100, -122.3450),
        (47.6050, -122.3200)
    ]
    
    optimizer = RouteOptimizerGA(locations, population_size=20, generations=50)
    optimal_path_indices = optimizer.run()