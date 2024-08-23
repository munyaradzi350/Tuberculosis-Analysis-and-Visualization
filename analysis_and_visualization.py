import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.stats import f_oneway
from sklearn.linear_model import LinearRegression
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import OLSInfluence

# Define the directories
normal_dir = '../data/normal'
tb_dir = '../data/tuberculosis'

# Function to load images
def load_images(img_dir, img_size=(128, 128)):
    try:
        image_files = [os.path.join(img_dir, file) for file in os.listdir(img_dir) if file.endswith('.png')]
        print("Image files found:", image_files)  # Debug print
        images = []
        for file in image_files:
            img = imread(file)
            if img.ndim == 3 and img.shape[2] == 3:
                img_resized = resize(img, img_size)
                images.append(img_resized)
            else:
                print(f"Ignoring {file} as it is not an RGB image.")
        return images
    except Exception as e:
        print(f"Error loading images from {img_dir}: {e}")
        return []

# Load images
normal_images = load_images(normal_dir)
tb_images = load_images(tb_dir)

# Check if images were loaded
if not normal_images:
    print("No normal images loaded.")
if not tb_images:
    print("No TB images loaded.")

# Visualize Sample Images
def plot_images(images, n, title, save_path):
    fig, axes = plt.subplots(1, n, figsize=(20, 20))
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.axis('off')
    fig.suptitle(title)
    plt.savefig(save_path)  # Save figure before showing
    plt.show()

# Plot a few normal images
plot_images(normal_images, 5, "Sample Normal Images", '../results/sample_normal_images.png')

# Plot a few TB images
plot_images(tb_images, 5, "Sample TB Images", '../results/sample_tb_images.png')

# Calculate Average Color Intensity
def get_avg_color_intensity(images):
    try:
        return [np.mean(rgb2gray(image)) for image in images]
    except Exception as e:
        print(f"Error calculating average intensity: {e}")
        return []

normal_intensity = get_avg_color_intensity(normal_images)
tb_intensity = get_avg_color_intensity(tb_images)

# Check if intensity calculations succeeded
if not normal_intensity:
    print("Failed to calculate intensity for normal images.")
if not tb_intensity:
    print("Failed to calculate intensity for TB images.")

# Combine and label the data
data = pd.DataFrame({
    'intensity': normal_intensity + tb_intensity,
    'type': ['Normal'] * len(normal_intensity) + ['TB'] * len(tb_intensity)
})

# One-way ANOVA
anova_result = f_oneway(normal_intensity, tb_intensity)
print("One-way ANOVA result:", anova_result)

# Add a size factor to the data frame
data['size'] = np.where(np.random.rand(len(data)) > 0.5, 'Small', 'Large')

# Two-way ANOVA
model = LinearRegression().fit(pd.get_dummies(data[['type', 'size']], drop_first=True), data['intensity'])
anova_result2 = anova_lm(model, typ=2)
print("Two-way ANOVA result:")
print(anova_result2)

# Cook's Distance
influence = OLSInfluence(model)
cooks_d = influence.cooks_distance[0]

# Plot Cook's Distance
plt.figure(figsize=(10, 6))
plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
plt.title("Cook's Distance")
plt.ylabel("Cook's Distance")
plt.xlabel("Observation Index")
plt.savefig('../results/cooks_distance.png')
plt.show()

# Histogram
print("Before plotting histogram")
plt.figure(figsize=(10, 6))
sns.histplot(data, x='intensity', hue='type', element='step', stat='density', common_norm=False)
plt.title("Histogram of Color Intensities")
plt.xlabel("Average Intensity")
plt.ylabel("Density")
plt.savefig('../results/histogram.png')
plt.show()
print("After plotting histogram")


# Line Graph
data_sorted = data.sort_values(by='intensity').reset_index(drop=True)
plt.figure(figsize=(10, 6))
for label, df in data_sorted.groupby('type'):
    plt.plot(df.index, df['intensity'], label=label)
plt.title("Line Graph of Color Intensities")
plt.xlabel("Image Index")
plt.ylabel("Average Intensity")
plt.legend()
plt.savefig('../results/line_graph.png')
plt.show()

# Bar Graph
avg_intensity = data.groupby('type')['intensity'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='type', y='intensity', data=avg_intensity)
plt.title("Bar Graph of Average Color Intensities")
plt.xlabel("Type")
plt.ylabel("Average Intensity")
plt.savefig('../results/bar_graph.png')
plt.show()

# Pie Chart
type_count = data['type'].value_counts().reset_index()
type_count.columns = ['type', 'count']
plt.figure(figsize=(8, 8))
plt.pie(type_count['count'], labels=type_count['type'], autopct='%1.1f%%', startangle=140)
plt.title("Pie Chart of Image Types")
plt.savefig('../results/pie_chart.png')
plt.show()
