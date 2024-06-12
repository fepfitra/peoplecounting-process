from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib
import os
import cv2
import random
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
import numpy as np

def load_images(images_folder):
    """
    Load images from the specified folder.

    Args:
        images_folder (str): The path to the folder containing images.

    Returns:
        list: A list of loaded images as NumPy arrays.
    """
    images = []
    # Iterate over the files in the specified folder
    for filename in os.listdir(images_folder):
        # Check if the file has a .jpg or .png extension
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            # Read the image using OpenCV
            img = cv2.imread(os.path.join(images_folder, filename))
            # Check if the image was read successfully
            if img is not None:
                # Convert the image to grayscale
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Append the loaded image to the list of images
                images.append(gray_img)
    return images


import albumentations as A


transform = A.Compose([
    # A.Crop(always_apply=False, p=0.5, x_min=0, y_min=0, x_max=140, y_max=140),
    A.HorizontalFlip(p=0.5),
    A.Rotate(always_apply=False, p=0.5, limit=(-45, 45), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False),
    A.Resize(always_apply=False, p=1, height=64, width=64, interpolation=0),
])

def apply_transform(image):
    transformed = transform(image=image)['image']
    return transformed

transform = A.Compose([
    A.Equalize(always_apply=False, p=1.0, mode='cv', by_channels=True, mask=None, mask_params=()),
    A.GaussianBlur(always_apply=False, p=1.0, blur_limit=(3, 5), sigma_limit=(0.0, 0)),
    A.Resize(always_apply=False, p=1, height=64, width=64, interpolation=0),
])

def apply_transform2(image):
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    transformed = transform(image=image)['image']
    return transformed

def hog_features(image, orient = 9, pix_per_cell = 8, cell_per_block = 2):
    # Convert the image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to the desired input size for HOG feature extraction
    #resized_image = resize_image(image, resize_width, resize_height)

    # Extract HOG features using the skimage hog function
    features = hog(image, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys', transform_sqrt=True,
                   feature_vector=True)

    # Apply histogram equalization to improve feature visibility
    features = exposure.equalize_hist(features)
    return features

def lbp_features(img, radius=1):
    # Convert the image to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Check the dimensions of the grayscale image
    if len(img.shape) != 2:
        # raise ValueError("Converted grayscale image is not 2-dimensional")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Parameters for LBP
    n_points = 8 * radius
    lbp_image = local_binary_pattern(img, n_points, radius, method='uniform')

    # Calculate the histogram
    # bins = int(lbp_image.max() + 1)
    bins = 10
    hist, _ = np.histogram(lbp_image.ravel(), bins=bins, range=(0, bins))

    return hist

from sklearn.preprocessing import MinMaxScaler
def normalize(feature:list)->list:
  scaler = MinMaxScaler()
  feature = np.array(feature).reshape(-1, 1)  # Pastikan feature adalah numpy array dan dua dimensi
  normalized_feature = scaler.fit_transform(feature)
  return normalized_feature.flatten()

def seq(img):
    if len(img.shape) == 3:  # Cek jika gambar berwarna
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img = blur(img)
    # img = scretchHist(img)
    # img = equalizeHist(img)
    img = apply_transform2(img)

    # Ekstraksi fitur HOG
    hog_feat = hog_features(img)
    # Ekstraksi fitur LBP
    lbp_feat = lbp_features(img)

    # Normalisasi fitur HOG
    normalized_hog_features = normalize(hog_feat)
    # Normalisasi fitur LBP
    normalized_lbp_features = normalize(lbp_feat)

    # Gabungkan fitur HOG dan LBP
    combined_features = np.concatenate((normalized_hog_features, normalized_lbp_features))

    #  # Pastikan panjang fitur konsisten
    # if len(combined_features) > target_length:
    #     combined_features = combined_features[:target_length]  # Potong jika terlalu panjang
    # elif len(combined_features) < target_length:
    #     padding = np.zeros(target_length - len(combined_features))
    #     combined_features = np.concatenate((combined_features, padding))

    return combined_features.flatten()


def sliding_window2(image, stepSize, windowSize):
    """
    Generate sliding windows over an image.

    Args:
        image (numpy.ndarray): The input image to slide the windows over.
        stepSize (int): The step size for moving the sliding window.
        windowSize (tuple): A tuple representing the size of the sliding window in (width, height).

    Yields:
        tuple: A tuple containing the coordinates (x, y) of the top-left corner of the window
               and the cropped window as a NumPy array.
    """
    for y in range(0, image.shape[0] - windowSize[1], stepSize):
        for x in range(0, image.shape[1] - windowSize[0], stepSize):
            # Yield the window's top-left coordinates and the cropped window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def hard_negative_mining2(svm, negative_images, stepSize, windowSize = (64, 64)):
    """
    Perform hard negative mining using a trained SVM classifier.

    Args:
        svm (object): A trained SVM classifier.
        negative_images (list): A list of negative images to mine hard negatives from.
        stepSize (int): The step size for moving the sliding window.

    Returns:
        list: A list of hard negative windows as NumPy arrays.
    """
    hard_negatives = []
    # Iterate over each negative image
    for image in negative_images:
        # Slide the window over the image
        for (x, y, window) in sliding_window2(image, stepSize, windowSize):
            # Extract HOG features from the window

            features = seq(window)
            # Make a prediction using the trained SVM
            pred = svm.predict([features])
            # If the prediction is positive, consider the window as a hard negative
            if pred[0] == 1:
                hard_negatives.append(window)
    return hard_negatives

import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import joblib
import os
import cv2
import random
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
import numpy as np

def non_max_suppression2(boxes, overlap_threshold):
    """
    Apply non-maximum suppression to a list of bounding boxes.

    Args:
        boxes (numpy.ndarray): An array of bounding boxes in [x1, y1, x2, y2] format.
        overlap_threshold (float): The threshold for considering two boxes as overlapping.

    Returns:
        numpy.ndarray: An array of selected bounding boxes after non-maximum suppression.
    """
    if len(boxes) == 0:
        return []
    # Convert bounding box coordinates to float
    boxes = boxes.astype("float")
    # Initialize the list of selected bounding boxes
    selected_boxes = []
    # Get coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # Calculate area of each bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # Sort the bounding box indexes based on y2 coordinate
    indexes = np.argsort(y2)
    while len(indexes) > 0:
        # Take the last bounding box in the index list and add it to selected_boxes
        last = len(indexes) - 1
        i = indexes[last]
        selected_boxes.append(i)
        # Calculate the intersection bounding box coordinates (maximum x1 and y1 among bounding boxes)
        xx1 = np.maximum(x1[i], x1[indexes[:last]])
        yy1 = np.maximum(y1[i], y1[indexes[:last]])
        xx2 = np.minimum(x2[i], x2[indexes[:last]])
        yy2 = np.minimum(y2[i], y2[indexes[:last]])
        # Calculate width and height of intersection
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # Calculate intersection area and overlap ratio with respect to original bounding boxes' areas
        overlap = (w * h) / areas[indexes[:last]]
        # Remove indexes of elements that have overlap higher than the threshold
        indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
    # Return the selected bounding boxes
    return boxes[selected_boxes].astype("int")

class ObjectDetectionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, svm_with_hard_negatives, winSizes, stepSize, downscale, threshold):
        self.svm_with_hard_negatives = svm_with_hard_negatives
        self.winSizes = winSizes
        self.stepSize = stepSize
        self.downscale = downscale
        self.threshold = threshold

    def detect_objects(self, image, winSize):
        detections = []
        for scale in np.linspace(1.0, self.downscale, 5)[::-1]:
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(image, (int(image.shape[1] / scale), int(image.shape[0] / scale)))

            for (x, y, window) in sliding_window2(resized_image, self.stepSize, winSize):
                if window.shape[0] != winSize[1] or window.shape[1] != winSize[0]:
                    continue
                features = seq(window)
                # if len(features) < 10414:
                #     continue
                confidence = self.svm_with_hard_negatives.decision_function([features])[0]
                print(f"processing {x} {y} {resized_image.shape} {winSize}, scale {scale}, confidence {confidence}")

                if confidence >= self.threshold:
                    x_coord = int(x * scale)
                    y_coord = int(y * scale)
                    w = int(winSize[0] * scale)
                    h = int(winSize[1] * scale)
                    detections.append((x_coord, y_coord, x_coord + w, y_coord + h))
        return detections

    def transform(self, X, y=None):
        all_detections = []
        for winSize in self.winSizes:
            detections = self.detect_objects(X, winSize)
            all_detections.extend(detections)
        # nms_detections = non_max_suppression2(np.array(all_detections), overlap_threshold=0.2)
        return all_detections


# if len(nms_detections) > 0:
#     bounding_box_list = nms_detections.tolist()
#     print(bounding_box_list)
# else:
#     print("NO FACES DETECTED IN THE IMAGE")
