import os
import pandas as pd

# Set the path to your dataset
dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data')

# Get the list of directories
normal_path = os.path.join(dataset_path, 'normal')
tb_path = os.path.join(dataset_path, 'tuberculosis')

# Get the list of all files in each directory
normal_images = os.listdir(normal_path)
tb_images = os.listdir(tb_path)

print(f'Number of normal images: {len(normal_images)}')
print(f'Number of tuberculosis images: {len(tb_images)}')

# Create a DataFrame with image paths and labels
data = []

for img in normal_images:
    data.append((os.path.join(normal_path, img), 'Normal'))
    
for img in tb_images:
    data.append((os.path.join(tb_path, img), 'Tuberculosis'))
    
df = pd.DataFrame(data, columns=['Image_Path', 'Label'])

# Save the DataFrame to a CSV file for further use
output_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'image_data.csv')
df.to_csv(output_path, index=False)
print(f'DataFrame saved to {output_path}')
