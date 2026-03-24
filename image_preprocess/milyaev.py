import os
import sys
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_niblack
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import img_as_float
import numpy as np
import maxflow
from preprocess_utils import ImageOutputSaver


def compute_weights(diff, spatial_dist, lambda_, sigma_g, sigma_c):
    return lambda_ * np.exp(-spatial_dist/(2*sigma_g**2) - diff/(2*sigma_c**2))


def milyaev_binarize(img):
    """
        Implements the binarization algorithm proposed in https://ieeexplore.ieee.org/abstract/document/6628598
    """

    # Cleaning image and converting to grayscale
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sigma = estimate_sigma(img_as_float(gray))
    sigmaColor = sigma * 255
    sigmaSpace = max(5, int(min(gray.shape) * 0.01))
    bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

    # Step 1 Apply niblack with small window
    H, W = gray.shape
    window_size = int(min(H, W) * 0.4)
    if window_size % 2 == 0:
        window_size += 1
    niblack_threshold = threshold_niblack(bilateral, window_size)
    niblack_boolean_mask = bilateral > niblack_threshold
    niblack = (niblack_boolean_mask * 255)

    # Step 2 Apply the normalization of absolute value of the image's laplacian
    laplacian = cv2.Laplacian(bilateral, cv2.CV_64F)
    norm_lap = cv2.normalize(abs(laplacian), None, 0, 255, cv2.NORM_MINMAX)

    def run_graph_cut(bilateral_input, img_input, mask):
        # Step 3 Energy function optimization
        # e_local term
        norm_lap_max_value = norm_lap/255.0
        cost_text = np.where(mask, 1-(0.5+(norm_lap_max_value/2)), 0.5+(norm_lap_max_value/2)) # e_local value if is text (1)
        cost_bg = np.where(mask, 0.5+(norm_lap_max_value/2), 1-(0.5+(norm_lap_max_value/2))) # e_local value if is bg (0)

        # e_smooth term
        sigma_g = 12 # constant from paper
        sigma_c = 0.02 # constant from paper
        lambda_ = 2 # constant from paper

        # 8-neighbours color distances
        # abs(ci-cj)^2
        img_norm = img_input.astype(float) / 255.0
        diff_h = np.sum((img_norm[:, :-1, :].astype(float) - img_norm[:, 1:, :].astype(float)) ** 2, axis=2) # Horizontal difference
        diff_v = np.sum((img_norm[:-1, :, :].astype(float) - img_norm[1:, :, :].astype(float)) ** 2, axis=2) # Vertical difference
        diff_top_left_bot_right = np.sum((img_norm[:-1, :-1, :].astype(float) - img_norm[1:, 1:, :].astype(float)) ** 2, axis=2) # Diagonal difference (top-left to bottom-right)
        diff_top_right_bot_left = np.sum((img_norm[:-1, 1:, :].astype(float) - img_norm[1:, :-1, :].astype(float)) ** 2, axis=2) # Diagonal difference (top-right to bottom-left)

        # Graph's weights
        weights_h = compute_weights(diff_h, 1, lambda_, sigma_g, sigma_c)
        weights_v = compute_weights(diff_v, 1, lambda_, sigma_g, sigma_c)
        weights_top_left_bot_right = compute_weights(diff_top_left_bot_right, 2, lambda_, sigma_g, sigma_c)
        weights_top_right_bot_left = compute_weights(diff_top_right_bot_left, 2, lambda_, sigma_g, sigma_c)

        structure_h = np.array([[0,0,0],
                                [0,0,1],
                                [0,0,0]
                                ])
        structure_v = np.array([[0,0,0],
                                [0,0,0],
                                [0,1,0]
                                ])
        structure_top_left_bot_right = np.array([
                                [0,0,0],
                                [0,0,0],
                                [0,0,1]
                                ])
        structure_top_right_bot_left = np.array([
                                [0,0,0],
                                [0,0,0],
                                [1,0,0]
                                ])
        
        # Fixing dimensions for graph
        h, w = bilateral_input.shape
        # Horizontal
        weights_h_full = np.zeros((h, w))
        weights_h_full[:, :-1] = weights_h

        # Vertical
        weights_v_full = np.zeros((h, w))
        weights_v_full[:-1, :] = weights_v

        # Diagonal
        weights_d1_full = np.zeros((h, w))
        weights_d1_full[:-1, :-1] = weights_top_left_bot_right

        # Diagonal
        weights_d2_full = np.zeros((h, w))
        weights_d2_full[:-1, 1:] = weights_top_right_bot_left

        # Graph
        graph = maxflow.Graph[float](h*w, 2*h*w) 
        nodeids = graph.add_grid_nodes((h, w)) # Adding nodes
        graph.add_grid_tedges(nodeids, cost_bg, cost_text) # Adding terminal edges

        # Adding non terminal edges
        graph.add_grid_edges(nodeids, weights_h_full, structure_h, symmetric=True)
        graph.add_grid_edges(nodeids, weights_v_full, structure_v, symmetric=True)
        graph.add_grid_edges(nodeids, weights_d1_full, structure_top_left_bot_right, symmetric=True)
        graph.add_grid_edges(nodeids, weights_d2_full, structure_top_right_bot_left, symmetric=True)

        # Execute maxflow
        graph.maxflow()

        return graph.get_grid_segments(nodeids).astype(np.uint8) * 255 # Getting the segments False -> Source (Background), True -> Sink (Text)

    # Run for dark text on light background
    result_normal = run_graph_cut(bilateral, img, niblack_boolean_mask)

    # Run for light text on dark background (inverted)
    bilateral_inv = 255 - bilateral
    niblack_threshold_inv = threshold_niblack(bilateral_inv, 5)
    niblack_boolean_mask_inv = bilateral_inv > niblack_threshold_inv
    niblack_inv = (niblack_boolean_mask_inv * 255)
    result_inverted = run_graph_cut(bilateral_inv, 255 - img, niblack_boolean_mask_inv)

    steps = [
        (original, "1. Original"),
        (gray, "2. Grayscale"),
        (bilateral, "3. Bilateral"),
        (niblack, "4a. Niblack"),
        (niblack_inv, "4b. Niblack (inverted)"),
        (norm_lap, "5. Laplacian"),
        (result_normal, "6a. Result (dark text)"),
        (result_inverted, "6b. Result (light text)"),
    ]

    return result_normal, result_inverted, steps


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])

    _, _, steps = milyaev_binarize(img)

    filename = os.path.basename(sys.argv[1])
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    saver = ImageOutputSaver("output")
    saver.save_mosaic(steps, filename, script_name)