import numpy as np
import matplotlib.pyplot as plt
#import imageio
from PIL import Image

# Parameters for the grid and diffusion
nx, ny = 100, 100  # Grid size
D = 1.0  # Diffusion coefficient
hx, hy = 1.0, 1.0  # Grid spacing
dt = 0.1  # Time step
timesteps = 1000  # Number of timesteps
Pe = 1  # Set desired Peclet number

# Define the source in the center of the grid
source_strength = 100.0

# Compute velocity v_x based on Peclet number
v_x = (Pe * D) / hx
v_y = 0  # Keep velocity in y-direction zero

# Print calculated v_x
print(f"Calculated v_x based on Pe={Pe}: {v_x}")

# Initialize the concentration grid
c = np.zeros((nx, ny))

# Source position
source_x, source_y = nx // 2, ny // 2

# Apply source in the center initially
c[source_x, source_y] = source_strength

# Prepare to save frames for the GIF
frames = []


# Function to compute Pe and check for stability
def compute_Pe(v, h, D):
    return v * h / D


# Updated code with buffer_rgba
for t in range(timesteps):
    # Copy the grid to avoid overwriting
    c_new = c.copy()

    # Loop over the grid points
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            # Update concentration using diffusion and advection (in both x and y directions)
            c_new[i, j] = (
                    c[i, j]
                    + D * dt / hx ** 2 * (c[i + 1, j] - 2 * c[i, j] + c[i - 1, j])  # Diffusion in x-direction
                    + D * dt / hy ** 2 * (c[i, j + 1] - 2 * c[i, j] + c[i, j - 1])  # Diffusion in y-direction
                    - v_x * dt / (2 * hx) * (c[i + 1, j] - c[i - 1, j])  # Advection in x-direction
                    - v_y * dt / (2 * hy) * (c[i, j + 1] - c[i, j - 1])  # Advection in y-direction
            )

    # Reapply the source concentration to ensure it remains constant
    c_new[source_x, source_y] = source_strength

    # Update the grid
    c = c_new

    # Visualization every 5 timesteps
    if t % 50 == 0:
        print(t)
        fig, ax = plt.subplots()
        im = ax.imshow(c, cmap="hot", interpolation="nearest")

        # Title with Peclet number
        plt.title(f"Time step: {t}, Pe = {Pe}")

        # Add legend with h, v_x, and D
        legend_text = f"h_x = {hx}, v_x = {v_x:.2f}, D = {D}"
        plt.legend([legend_text], loc="upper right", fontsize="small", frameon=False)

        plt.colorbar(im, label="Concentration")

        # Save the current plot as an image in memory
        fig.canvas.draw()

        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')  # Use buffer_rgba instead of tostring_rgb
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA has 4 channels

        # Convert to PIL Image and append to frames
        frames.append(Image.fromarray(image))
        plt.close()

# Save the frames as a GIF
frames[0].save('diffusion_simulation_advection.gif', format='GIF', append_images=frames[1:], save_all=True,
               duration=100, loop=0)

print("GIF saved as 'diffusion_simulation_advection.gif'")