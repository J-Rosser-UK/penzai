import matplotlib.pyplot as plt

# Specify the path to your file
filename = "losses.txt"

# Initialize lists to store step and loss values
steps = []
losses = []

# Read the file and parse the data
with open(filename, "r") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue  # skip empty lines
        try:
            # Each line is expected to have two comma-separated values: step and loss
            step_str, loss_str = line.split(",")
            step = float(step_str)
            loss = float(loss_str)
            steps.append(step)
            losses.append(loss)
        except ValueError as e:
            print(f"Error parsing line '{line}': {e}")
            continue

# Create a figure and plot the data
plt.figure(figsize=(8, 6))
plt.plot(steps, losses, marker="o", linestyle="-", color="b")

# Label the axes and add a title
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss vs Step")

# Set both x and y axes to logarithmic scale
plt.xscale("log")
plt.yscale("log")

# Add grid lines for better readability on log scales
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Display the plot
plt.show()
