import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))  # Just a sample function

# Create contour plot
plt.contour(X, Y, Z, levels=2)  # Draw 10 contour lines
plt.colorbar()  # Add a colorbar to see the corresponding Z values
plt.title("Contour plot of Z")
plt.show()



