import numpy as np
from gurobipy import Model, GRB, quicksum
import pandas as pd

# Set random seed for reproducibility
np.random.seed(10)

# Load the data from the Excel file
file_path = 'Data.xlsx'
data = pd.read_excel(file_path)

# Extract coordinates and waste generation
demand_nodes = data[['Latitude', 'Longitude']].values
waste_generation = data['Waste Generation (kg/day)'].values

# Coordinates for parking and processing nodes
parking_nodes = np.array([[21.7963850, 39.1370496]])
processing_nodes = np.array([[21.9128747, 39.1201999]])

# Append node's coordinates in a single matrix
nodes = np.vstack((demand_nodes, parking_nodes, processing_nodes))
number_nodes = nodes.shape[0]

# Extracting each type of node indices
number_demand_nodes = len(demand_nodes)
number_parking_nodes = len(parking_nodes)
number_processing_nodes = len(processing_nodes)
demand_indices = range(number_demand_nodes)
parking_indices = range(number_demand_nodes, number_demand_nodes + number_parking_nodes)
processing_indices = range(number_demand_nodes + number_parking_nodes,
                           number_demand_nodes + number_parking_nodes + number_processing_nodes)

# Number of vehicles
number_vehicles = 2
# Constant speed for all vehicles in km/h
speed_kmh = 40
# Number of minutes per time unit
minutes_per_time_unit = 5
# Speed in km/time unit
speed = speed_kmh / (60 / minutes_per_time_unit)
vehicles_speed = np.full(number_vehicles, speed)
# Distance matrix
distance_matrix = np.linalg.norm(nodes[:, np.newaxis] - nodes[np.newaxis, :], axis = 2)
# Distance to kilometers
distance_matrix = distance_matrix * 110.574

# Starting parking nodes (sampling from parking indices)
starting_nodes = np.random.choice(parking_indices, size = number_vehicles, replace = True)

# Time horizon across 8 hours
final_time = int(12 * (60 / minutes_per_time_unit))

# Traveling time matrix
traveling_times = np.zeros((number_nodes, number_nodes, number_vehicles), dtype = int)
for vehicle in range(number_vehicles):
    traveling_times[:, :, vehicle] = distance_matrix / vehicles_speed[vehicle]

# Stating that the minimum traveling time is one time unit at the average speed
traveling_times[traveling_times == 0] = 1
# Setting the traveling time of the diagonal to 0
for vehicle in range(number_vehicles):
    np.fill_diagonal(traveling_times[:, :, vehicle], 0)

# Base rate of waste generation for each demand node (initially given in kg/days)
alpha = waste_generation / (24 * 60 / minutes_per_time_unit)

# Exposure risk factor parameters for each demand node
beta = np.random.uniform(0, 1, number_demand_nodes)

# Maximum vehicle capacities of 1000 kg
vehicles_capacities = np.full(number_vehicles, 1000)

# Traveling costs of 2$ per minute
traveling_costs = np.full(number_vehicles, 2 * minutes_per_time_unit)

# Create the optimization model
model = Model('MedicalWasteCollection')

# Decision variables
x = model.addVars(number_nodes, number_nodes, number_vehicles, final_time, vtype = GRB.BINARY, name = 'x')
W = model.addVars(number_demand_nodes, final_time, vtype = GRB.CONTINUOUS, lb = 0, name = 'W')
l = model.addVars(number_vehicles, final_time, vtype = GRB.CONTINUOUS, lb = 0, name = 'l')
WX = model.addVars(number_demand_nodes, number_vehicles, final_time, vtype = GRB.CONTINUOUS, lb = 0, name = 'WX')

# Objective functions
f1 = quicksum(traveling_costs[v] * quicksum(
    traveling_times[i, j, v] * x[i, j, v, t] for i in range(number_nodes) for j in range(number_nodes) for t in
    range(final_time)) for v in range(number_vehicles))

f2 = quicksum(beta[i] * quicksum(W[i, t] for t in range(final_time)) for i in range(number_demand_nodes))

# Calculating maximum values for both objective functions for normalization

# Calculating for distance objective function by having all vehicles go the furthest node from their starting node, and
# back, until the final time is reached
f1_max = 0
for v in range(number_vehicles):
    # Find the furthest node from the starting node
    furthest_node = np.argmax(distance_matrix[starting_nodes[v], :])
    # Calculate the time it takes to go to the furthest
    time_to_furthest_node = traveling_times[starting_nodes[v], furthest_node, v]
    # Calculate the number of times the vehicle can go to the furthest node and back
    number_trips = final_time // (2 * time_to_furthest_node)
    # Calculate the cost of the vehicle going to the furthest node and back
    f1_max += traveling_costs[v] * number_trips * 2 * time_to_furthest_node

# Calculating for exposure risk objective function by having all demand nodes generate waste all the time and not being
# collected
f2_max = sum(beta[i] * alpha[i] * final_time for i in range(number_demand_nodes))

# Assigning normalized values to the objective functions
model.setObjective(f1 / f1_max + f2 / f2_max, GRB.MINIMIZE)

# Constraints

M = (number_nodes + 1) * (number_nodes + 1)

for t in range(final_time):
    for v in range(number_vehicles):
        # Constraint 2: No vehicle can traverse more than one edge simultaneously
        model.addConstr(quicksum(x[i, j, v, t] for i in range(number_nodes) for j in range(number_nodes)) <= 1,
                        f'cons2_{v}_{t}')
        # Constraint 14: Maximum vehicle load at each time step
        model.addConstr(l[v, t] <= vehicles_capacities[v], f'cons13_{v}_{t}')
        if t < final_time - 1:
            # Constraint 13: Load adjustment at each vehicle over time
            model.addConstr(l[v, t] -
                            M * quicksum(x[p_c, i, v, t + 1]
                                         for p_c in processing_indices
                                         for i in range(number_nodes)) +
                            quicksum(WX[p_d, v, t + 1] for p_d in demand_indices) <= l[v, t + 1],
                            f'cons13{v}_{t}')
        if t > 0:
            for i in demand_indices:
                # Constraint 15: Waste collection at each demand node
                model.addConstr(
                    WX[i, v, t] >= W[i, t - 1] + alpha[i] - M * (
                            1 - quicksum(x[i, j, v, t] for j in range(number_nodes))),
                    f'cons15_{i}_{v}_{t}')

for v in range(number_vehicles):
    # Constraint 1: Each vehicle must start its route at its starting parking node
    model.addConstr(quicksum(x[starting_nodes[v], j, v, 0] for j in demand_indices) == 1, f'cons1_{v}')
    # Constraint 8: Each vehicle must visit one parking node immediately after visiting a processing node
    model.addConstr(quicksum(x[p_c, p_p, v, t] for p_c in processing_indices for p_p in parking_indices
                             for t in range(final_time)) == 1, f'cons8_{v}')
    # Constraint 12: Initial load of each vehicle
    model.addConstr(l[v, 0] == 0, f'cons12_{v}')
    for i in range(number_nodes):
        # Constraint 6: No vehicle can remain stationary at any node
        model.addConstr(quicksum(x[i, i, v, t] for t in range(final_time)) == 0, f'cons6_{i}_{v}')
        for t in range(final_time):
            if t > 0 and i not in parking_indices:
                # Constraint 5: Vehicles can only depart from nodes they have previously arrived at (i instead of j)
                model.addConstr(quicksum(x[i, k, v, t] for k in range(number_nodes) if k != i) <=
                                quicksum(x[h, i, v, t - traveling_times[h, i, v]] for h in range(number_nodes)
                                         if h != i and t >= traveling_times[h, i, v]), f'cons5_{i}_{v}_{t}')
        for j in range(number_nodes):
            if i != j:
                travel_time = traveling_times[i, j, v]
                # Constraint 3: Vehicles must respect travel times between nodes
                for t in range(final_time - travel_time):
                    model.addConstr(x[i, j, v, t] + quicksum(x[j, k, v, t_sum] for k in range(number_nodes)
                                                             for t_sum in range(t + 1, t + travel_time - 1)
                                                             if k != j) <= 1, f'cons3_{i}_{j}_{v}_{t}')
                    if j not in parking_indices:
                        # Constraint 4: Immediate departure from parking nodes
                        model.addConstr(
                            x[i, j, v, t] <= quicksum(x[j, k, v, t + travel_time]
                                                      for k in range(number_nodes)
                                                      if k != j), f'cons4_{i}_{j}_{v}_{t}')

for j in demand_indices:
    # Constraint 7: Each demand node must be visited at least once
    model.addConstr(quicksum(x[i, j, v, t] for i in range(number_nodes) for v in range(number_vehicles)
                             for t in range(final_time)) >= 1, f'cons7_{j}')
    # Constraint 10: Initial amount of waste at each demand node
    model.addConstr(W[j, 0] == 0, f'cons10_{j}')
    for t in range(final_time - 1):
        # Constraint 11: Amount of waste at each demand node over time
        model.addConstr(W[j, t + 1] >= W[j, t] + alpha[j] - M * quicksum(x[j, i, v, t + 1]
                                                                         for i in range(number_nodes)
                                                                         for v in range(number_vehicles)),
                        f'cons11_{j}_{t}')

# Constraint 9: Each vehicle's route ends once it has visited a parking node
for p_p in parking_indices:
    for v in range(number_vehicles):
        for t_prime in range(final_time - 1):
            model.addConstr(M * (1 - quicksum(x[i, p_p, v, t_prime] for i in range(number_nodes))) >=
                            quicksum(x[i, j, v, t] for i in range(number_nodes) for j in range(number_nodes)
                                     for t in range(t_prime + 1, final_time)), f'cons9_{p_p}_{v}_{t_prime}')

print('Model created successfully')

# Optimize the model
model.setParam('Method', -1)
model.optimize()

# Check if the model is solved successfully
if model.status == GRB.OPTIMAL:
    print('Optimal solution found')

    # For each vehicle, print the route and the time it arrives at each node
    for v in range(number_vehicles):
        print(f'Vehicle {v + 1} route:')
        current_node = starting_nodes[v]
        for t in range(final_time):
            for j in range(number_nodes):
                if x[current_node, j, v, t].x > 0.5:
                    # Find hour of the day and minute of the hour (considering the time unit)
                    hour = t // (60 / minutes_per_time_unit)
                    minute = (t % (60 / minutes_per_time_unit)) * minutes_per_time_unit
                    print(f'Departured to Node {j + 1} at {hour}:{minute}')
                    current_node = j
                    break

        print('')


else:
    print('No optimal solution found')
