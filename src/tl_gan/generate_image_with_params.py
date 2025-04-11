""" module to generate images with specific feature parameters """

import os
import sys
import numpy as np
import tensorflow as tf
import pickle
import glob
import logging
from PIL import Image
import src.tl_gan.feature_axis as feature_axis
import src.tl_gan.generate_image as generate_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
    logger.info("Starting image generation with features: %s", feature_values)
    if locked_features:
        logger.info("Locked features: %s", locked_features)
    
    # Load feature directions
    logger.info("Loading feature directions...")
    path_feature_direction = './asset_results/pg_gan_celeba_feature_direction_40'
    pathfile_feature_direction = glob.glob(os.path.join(path_feature_direction, 'feature_direction_*.pkl'))[-1]
    logger.debug("Using feature direction file: %s", pathfile_feature_direction)
    
    with open(pathfile_feature_direction, 'rb') as f:
        feature_direction_name = pickle.load(f)
    
    feature_direction = feature_direction_name['direction']
    feature_names = feature_direction_name['name']
    logger.info("Loaded %d feature directions", len(feature_names))
    
    # Load GAN model
    logger.info("Loading GAN model...")
    path_pg_gan_code = './src/model/pggan'
    path_model = './asset_model/karras2018iclr-celebahq-1024x1024.pkl'
    sys.path.append(path_pg_gan_code)
    
    sess = tf.InteractiveSession()
    with open(path_model, 'rb') as file:
        G, D, Gs = pickle.load(file)
    logger.info("GAN model loaded successfully")
    
    # Initialize latent vector
    logger.info("Initializing latent vector...")
    latents = np.random.randn(1, *Gs.input_shapes[0][1:])
    
    # Handle locked features
    if locked_features is None:
        locked_features = []
    locked_indices = [feature_names.index(f) for f in locked_features if f in feature_names]
    if locked_indices:
        logger.info("Locking %d features", len(locked_indices))
    
    # Disentangle feature directions
    logger.info("Disentangling feature directions...")
    feature_direction_disentangled = feature_axis.disentangle_feature_axis_by_idx(
        feature_direction, 
        idx_base=np.array(locked_indices)
    )
    
    # Apply feature values
    logger.info("Applying feature modifications...")
    for feature_name, value in feature_values.items():
        if feature_name in feature_names:
            idx = feature_names.index(feature_name)
            latents += feature_direction_disentangled[:, idx] * value
            logger.debug("Applied %s: %.2f", feature_name, value)
        else:
            logger.warning("Feature '%s' not found in available features", feature_name)
    
    # Generate image
    logger.info("Generating image...")
    image = generate_image.gen_single_img(z=latents[0], Gs=Gs)
    logger.info("Image generation complete")
    
    return image

def get_available_features():
    """
    Get list of available features that can be controlled
    
    Returns:
        list of feature names
    """
    logger.info("Retrieving available features...")
    path_feature_direction = './asset_results/pg_gan_celeba_feature_direction_40'
    pathfile_feature_direction = glob.glob(os.path.join(path_feature_direction, 'feature_direction_*.pkl'))[-1]
    
    with open(pathfile_feature_direction, 'rb') as f:
        feature_direction_name = pickle.load(f)
    
    features = feature_direction_name['name']
    logger.info("Found %d available features", len(features))
    return features

if __name__ == '__main__':
    # Print available features
    print("\nAvailable features:")
    print("-----------------")
    features = get_available_features()
    for i, feature in enumerate(features, 1):
        print(f"{i:2d}. {feature}")
    print("\n")

    # Example usage
    logger.info("Starting example image generation")
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
    output_path = 'generated_image.png'
    Image.fromarray(image).save(output_path)
    logger.info("Image saved to %s", output_path) 