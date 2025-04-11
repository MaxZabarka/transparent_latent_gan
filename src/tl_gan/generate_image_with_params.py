""" module to generate images with specific feature parameters """

import os
import sys
import numpy as np
import tensorflow as tf
import pickle
import glob
from PIL import Image
from src.tl_gan.feature_axis import disentangle_feature_axis_by_idx
from src.tl_gan.generate_image import gen_single_img

def generate_image_with_features(feature_values, locked_features=None):
    """
    Generate an image with specific feature values
    
    Args:
        feature_values: dict of feature names and their values
                       e.g. {'Male': 0.5, 'Beard': -0.3}
        locked_features: list of feature names to lock (optional)
    
    Returns:
        numpy array of the generated image
    """
    # Load feature directions
    path_feature_direction = './asset_results/pg_gan_celeba_feature_direction_40'
    pathfile_feature_direction = glob.glob(os.path.join(path_feature_direction, 'feature_direction_*.pkl'))[-1]
    
    with open(pathfile_feature_direction, 'rb') as f:
        feature_direction_name = pickle.load(f)
    
    feature_direction = feature_direction_name['direction']
    feature_names = feature_direction_name['name']
    
    # Load GAN model
    path_pg_gan_code = './src/model/pggan'
    path_model = './asset_model/karras2018iclr-celebahq-1024x1024.pkl'
    sys.path.append(path_pg_gan_code)
    
    sess = tf.InteractiveSession()
    with open(path_model, 'rb') as file:
        G, D, Gs = pickle.load(file)
    
    # Initialize latent vector
    latents = np.random.randn(1, *Gs.input_shapes[0][1:])
    
    # Handle locked features
    if locked_features is None:
        locked_features = []
    locked_indices = [feature_names.index(f) for f in locked_features if f in feature_names]
    
    # Disentangle feature directions
    feature_direction_disentangled = disentangle_feature_axis_by_idx(
        feature_direction, 
        idx_base=np.array(locked_indices)
    )
    
    # Apply feature values
    for feature_name, value in feature_values.items():
        if feature_name in feature_names:
            idx = feature_names.index(feature_name)
            latents += feature_direction_disentangled[:, idx] * value
    
    # Generate image
    image = gen_single_img(z=latents[0], Gs=Gs)
    
    return image

def get_available_features():
    """
    Get list of available features that can be controlled
    
    Returns:
        list of feature names
    """
    path_feature_direction = './asset_results/pg_gan_celeba_feature_direction_40'
    pathfile_feature_direction = glob.glob(os.path.join(path_feature_direction, 'feature_direction_*.pkl'))[-1]
    
    with open(pathfile_feature_direction, 'rb') as f:
        feature_direction_name = pickle.load(f)
    
    return feature_direction_name['name']

if __name__ == '__main__':
    # Example usage
    feature_values = {
        'Male': 0.5,        # More masculine
        'Beard': -0.3,      # Less beard
        'Smiling': 0.8,     # More smiling
        'Young': 0.4        # More young
    }

    # Optionally lock some features
    locked_features = ['Male']  # Lock the 'Male' feature while modifying others

    # Generate the image
    image = generate_image_with_features(feature_values, locked_features)

    # Save the image
    Image.fromarray(image).save('generated_image.png') 