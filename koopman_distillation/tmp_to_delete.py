import matplotlib.pyplot as plt
import pandas as pd

# Data
model_sizes = [20, 38, 84]
fid_ours = [10.2, 5.04, 4.89]
fid_get = [10.72, 8.00, 7.19]

# Create a DataFrame for plotting
df = pd.DataFrame({
    'Model Size (M)': model_sizes,
    'Ours': fid_ours,
    'GET': fid_get
})

# Plotting
plt.figure(figsize=(6, 4))
plt.plot(df['Model Size (M)'], df['Ours'], marker='o', linewidth=3.5, alpha=0.7, label='Ours', markersize=8,)
plt.plot(df['Model Size (M)'], df['GET'], marker='s', linewidth=3.5, alpha=0.7, label='GET', markersize=8,)

# Styling for publication
# plt.xlabel('Model Size (Millions)', fontsize=12)
# plt.ylabel('FID Score ↓', fontsize=12)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
# plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(model_sizes)  # Remove x-axis ticks
plt.yticks([11, 7, 4])  # Remove y-axis ticks

# Remove top and right spines
for spine in ['top', 'right']:
    plt.gca().spines[spine].set_visible(False)

plt.legend(fontsize=25)
plt.tight_layout()

# Save as high-resolution PDF
output_path = "fid_vs_model_size_plot.pdf"
plt.savefig(output_path, format='pdf', dpi=300)
plt.show()
