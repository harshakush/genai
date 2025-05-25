import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a figure and 3D axis
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Define the starting and ending points (A and B)
point_A = np.array([0, 0, 0])  # Starting point (e.g., Delhi)
point_B = np.array([800, 600, 0])  # Destination (e.g., Mumbai)

# Define the actual flight path with altitude changes
# The flight doesn't go in a straight line due to air traffic, wind, etc.
num_points = 50
t = np.linspace(0, 1, num_points)

# Create a curved path with altitude changes
flight_path = np.zeros((num_points, 3))
for i in range(num_points):
    # X and Y follow a curved path
    flight_path[i, 0] = point_A[0] + (point_B[0] - point_A[0]) * t[i]  # x-coordinate
    flight_path[i, 1] = point_A[1] + (point_B[1] - point_A[1]) * t[i] + 100 * np.sin(3 * np.pi * t[i])  # y-coordinate with deviation
    
    # Z (altitude) follows a typical flight profile: climb, cruise, descent
    if t[i] < 0.15:  # Climb phase
        flight_path[i, 2] = 10000 * (t[i] / 0.15)
    elif t[i] > 0.85:  # Descent phase
        flight_path[i, 2] = 10000 * (1 - (t[i] - 0.85) / 0.15)
    else:  # Cruise phase
        flight_path[i, 2] = 10000 + 500 * np.sin(np.pi * (t[i] - 0.15) / 0.7)  # Small altitude adjustments during cruise

# Calculate the direct vector from A to B
direct_vector = point_B - point_A

# Plot the direct path from A to B (ground projection)
ax.plot([point_A[0], point_B[0]], [point_A[1], point_B[1]], [0, 0], 'k--', linewidth=2, label='Direct Ground Path')

# Plot the actual flight path
ax.plot(flight_path[:, 0], flight_path[:, 1], flight_path[:, 2], 'r-', linewidth=3, label='Air India Flight Path')

# Plot the ground projection of the flight path
ax.plot(flight_path[:, 0], flight_path[:, 1], np.zeros_like(flight_path[:, 2]), 'g-', linewidth=2, alpha=0.5, label='Ground Projection')

# Plot vertical projections at a few points to show altitude
for i in range(0, num_points, 10):
    ax.plot([flight_path[i, 0], flight_path[i, 0]], 
            [flight_path[i, 1], flight_path[i, 1]], 
            [0, flight_path[i, 2]], 'b--', alpha=0.3)

# Mark the start and end points
ax.scatter(point_A[0], point_A[1], point_A[2], color='green', s=100, label='Departure (A)')
ax.scatter(point_B[0], point_B[1], point_B[2], color='red', s=100, label='Destination (B)')

# Add a current position of the aircraft (e.g., 60% through the journey)
current_pos_idx = int(0.6 * num_points)
current_pos = flight_path[current_pos_idx]
ax.scatter(current_pos[0], current_pos[1], current_pos[2], color='blue', s=200, marker='^', label='Current Position')

# Calculate and plot the velocity vector at the current position
if current_pos_idx < num_points - 1:
    velocity_vector = flight_path[current_pos_idx + 1] - flight_path[current_pos_idx]
    velocity_vector = velocity_vector / np.linalg.norm(velocity_vector) * 50  # Scale for visualization
    ax.quiver(current_pos[0], current_pos[1], current_pos[2], 
              velocity_vector[0], velocity_vector[1], velocity_vector[2], 
              color='blue', arrow_length_ratio=0.1, linewidth=2, label='Velocity Vector')

# Calculate the remaining direct vector to destination
remaining_vector = point_B - current_pos
remaining_vector_ground = np.array([remaining_vector[0], remaining_vector[1], 0])

# Plot the remaining direct vector to destination
ax.quiver(current_pos[0], current_pos[1], current_pos[2], 
          remaining_vector[0], remaining_vector[1], remaining_vector[2], 
          color='purple', arrow_length_ratio=0.1, linewidth=2, label='Direct Vector to B')

# Calculate and plot the projection of velocity onto the remaining direct vector
# First, normalize the remaining vector
remaining_vector_norm = remaining_vector / np.linalg.norm(remaining_vector)
# Calculate the dot product (scalar projection)
scalar_projection = np.dot(velocity_vector, remaining_vector_norm)
# Calculate the vector projection
vector_projection = scalar_projection * remaining_vector_norm

# Plot the projection
ax.quiver(current_pos[0], current_pos[1], current_pos[2], 
          vector_projection[0], vector_projection[1], vector_projection[2], 
          color='orange', arrow_length_ratio=0.1, linewidth=2, label='Velocity Projection')

# Set labels and title
ax.set_xlabel('X Distance (km)')
ax.set_ylabel('Y Distance (km)')
ax.set_zlabel('Altitude (m)')
ax.set_title('Air India Flight: 3D Trajectory with Vector Projections')

# Add text annotations
ax.text(point_A[0], point_A[1], point_A[2] + 500, 'Delhi', color='green')
ax.text(point_B[0], point_B[1], point_B[2] + 500, 'Mumbai', color='red')

# Add explanation of the projection
projection_efficiency = np.dot(velocity_vector, remaining_vector_norm) / np.linalg.norm(velocity_vector) * 100
ax.text2D(0.02, 0.05, f"Projection Efficiency: {projection_efficiency:.1f}%\n" +
                     "This shows how effectively the aircraft is\n" +
                     "moving toward its destination", 
         transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

# Set axis limits
ax.set_xlim(0, point_B[0] * 1.1)
ax.set_ylim(0, point_B[1] * 1.1)
ax.set_zlim(0, 12000)

# Add a legend
ax.legend(loc='upper right')

# Show the plot
plt.tight_layout()
plt.show()
