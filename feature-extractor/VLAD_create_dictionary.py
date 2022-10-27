import argparse
import numpy as np
from extractors import *
import logging
from os import access
from os import R_OK
from os import W_OK
from os.path import isfile
from os.path import isdir
import os
import sys
import time
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image_list", required=True, type=str,
                    help="Path to a file with paths to images to be extracted.")
parser.add_argument("-o", "--output_dir", required=True,
                    type=str, help="Path to an output folder.")
parser.add_argument("-s", "--size", required=True, type=int,
                    help="Size of a dictionary.")

def extract_sift(image_paths):
    import cv2 as cv
    sift = cv.SIFT_create()

    features = []
    for img_path in image_paths:
        img = cv.imread(img_path)
        gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        _, des = sift.detectAndCompute(gray,None)
        
        if des is not None:
            features.append(des)
    return np.concatenate(features)

def main(args):
    logging.basicConfig(
            format='%(levelname)s:\t%(message)s', level=logging.INFO)

    images_paths = []
    with open(args.image_list, "r") as file:
        images_paths = [line.rstrip() for line in file.readlines()]

    logging.debug("Image list read. Path sample: ")
    logging.debug(images_paths[:3])

    features = extract_sift(images_paths)

    features = features.reshape((-1, 128))

    logging.info(f"SIFT features size = {features.shape}")

    dictionary = KMeans(n_clusters=args.size).fit(features).cluster_centers_

    logging.info(f"Created dictionary, shape = {dictionary.shape}")
    
    with open(os.path.join(args.output_dir, f"VLAD_dict_{args.size}.npy"), 'wb') as f:
        np.save(f, dictionary)

    logging.info("DONE")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
