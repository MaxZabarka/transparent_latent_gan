""" module to generate iamges from pg-gan """

import os
import sys
import numpy as np
import tensorflow as tf
import PIL

path_pg_gan_code = './src/model/pggan'
path_model = './asset_model/karras2018iclr-celebahq-1024x1024.pkl'
sys.path.append(path_pg_gan_code)

len_z = 512
len_dummy = 0

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def gen_single_img(z=None, Gs=None):
    """
    function to generate image from noise
    :param z:  1D array, latent vector for generating images
    :param Gs: generator network of GAN
    :return:   one rgb image, H*W*3
    """
    logger.info("Starting image generation")

    if z is None:  # if input not given
        logger.debug("No latent vector provided, generating random vector")
        z = np.random.randn(len_z)
    
    if len(z.shape) == 1:
        logger.debug("Reshaping latent vector to 2D array")
        z = z[None, :]
        
    logger.debug("Creating dummy variables")
    dummy = np.zeros([z.shape[0], len_dummy])
    
    logger.info("Running generator network")
    images = Gs.run(z, dummy)
    
    logger.debug("Post-processing generated images")
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
    
    logger.info("Image generation complete")
    return images[0]


def save_img(img, pathfile):
    PIL.Image.fromarray(img, 'RGB').save(pathfile)