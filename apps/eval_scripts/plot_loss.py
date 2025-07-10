import matplotlib.pyplot as plt

# --- How to Use ---
# 1. Install matplotlib: pip install matplotlib
# 2. Replace the sample data below with your own data.
# 3. Run the script: python plot_loss.py

# Sample data (replace with your actual data)
# You would parse your log file to get these lists.
steps = [1, 2, 3, 4, 5]
loss_values = [7.479, 7.567, 7.278, 6.839, 6.016]

# Create the plot
plt.figure(figsize=(10, 6))  # Set the figure size for better readability
plt.plot(steps, loss_values, marker='o', linestyle='-')

# Add titles and labels for clarity
plt.title('Training Loss vs. Step', fontsize=16)
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Loss', fontsize=12)

# Add a grid for easier analysis
plt.grid(True)

# Display the plot
plt.savefig('training_loss_plot.png', dpi=300)
print("Plot saved as training_loss_plot.png")