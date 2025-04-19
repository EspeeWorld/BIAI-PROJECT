import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_excel('flight data.xlsx', sheet_name='Sheet1')
# df = pd.read_csv('flight data.csv', encoding='utf-8')
df.to_csv('new_file.csv', index=False)
df = df.fillna(0)
df = df.drop(index=0)
df = df.rename(columns={'Time (EDT)': 'Time'})

# Add new columns for upper and lower limit values
x = np.array(df['feet'])
log_x = np.log(x)
mn_x = np.mean(x)
n = len(x)

# Function to compute Y value
def compute_Y(x_value, n, mean, speed, epsilon=1e-6, delta=1e-6, alpha=1.0):
    """
    Computes the modified Y value based on the given parameters, ensuring convergence as speed -> 0.
    
    Parameters:
        x: float, the x-value in the formula
        n: int, a positive integer
        mean: float, mean value
        speed: float, speed variable
        epsilon: small positive constant to avoid log(0)
        delta: small positive constant to avoid division by zero
        alpha: decay factor for speed convergence

    Returns:
        - Computed Y value
    """
    log_term = np.log(x_value + epsilon)
    sqrt_term = np.sqrt(1 / (n + delta))
    speed_factor = np.exp(-alpha * speed)

    Y = 2 * (log_term * sqrt_term * speed_factor) ** 2 * mean
    return Y

# Following Heisenberg's Uncertainty Principle from quantum mechanics - it is impossible to simultaneously know both the exact position and the exact momentum (or speed) of an entity 
# (i.e. the more precisely you know its position, the less you know about its momentum (which relates to speed) and vice versa!
# Thus, speed is kept constant, since the safety emphasis here is the position of the object on a time series. 
speed = 1.0  

# Compute the lower and upper limits using the compute_Y function
Y = np.array([compute_Y(x_value, n, mn_x, speed) for x_value in x])

df['lower_limit'] = x - Y
df['upper_limit'] = x + Y
df['safe_zone'] = (x + Y) - (x - Y)

print(df.head())

# Plot safety boundaries over time
plt.figure(figsize=(10, 6))  # Set the figure size

plt.plot(df['Time'], df['feet'], label='Altitude (Feet)', color='green')
plt.plot(df['Time'], df['lower_limit'], label='Lower Limit', color='red', linestyle='--')
plt.plot(df['Time'], df['upper_limit'], label='Upper Limit', color='blue', linestyle='--')

# Add labels and a title
plt.xlabel('Time Series')
plt.ylabel('Altitude (Feet)')
plt.title('Altitude Over Time')

# Optionally add gridlines
#plt.grid(True)

# Add a legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

# Plot safety zone in 'feet' over time

plt.plot(df['feet'], df['safe_zone'], label='safe_zone', color='black')

# Add labels and a title
plt.xlabel('Feet')
plt.ylabel('Safe_zone')
plt.title('Safety Zone in "Feet" over Altitude')

# Add a legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()