import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.stats import f_oneway
from sklearn.linear_model import LinearRegression
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import OLSInfluence

def load_images(img_dir, img_size=(128, 128)):
    image_files = [os.path.join(img_dir, file) for file in os.listdir(img_dir) if file.endswith('.png')]
    images = [resize(imread(file), img_size) for file in image_files]
    return images

def get_avg_color_intensity(images):
    return [np.mean(rgb2gray(image)) for image in images]

def perform_anova(normal_intensity, tb_intensity):
    return f_oneway(normal_intensity, tb_intensity)

def perform_two_way_anova(data):
    model = LinearRegression().fit(pd.get_dummies(data[['type', 'size']], drop_first=True), data['intensity'])
    return anova_lm(model, typ=2)

def calculate_cooks_distance(model):
    influence = OLSInfluence(model)
    return influence.cooks_distance[0]
