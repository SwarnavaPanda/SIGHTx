'''import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.utils import resample
#from mymy import convert_tiff_to_jpg

def load_image_pixels(image_path):
    """Loads an image and reshapes it into (num_pixels, 3) array."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    return pixels


def compute_Wk(data, n_clusters):
    """Computes within-cluster dispersion for given number of clusters."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    Wk = 0
    for i in range(n_clusters):
        cluster_data = data[labels == i]
        if cluster_data.shape[0] > 0:
            distances = pairwise_distances(cluster_data, [cluster_centers[i]])
            Wk += np.sum(distances ** 2)
    return Wk


def gap_statistic(data, refs=10, max_k=10):
    """Computes gap statistics for k=1 to max_k"""
    shape = data.shape
    gaps = []
    Wks = []
    Wkbs = []
    ks = np.arange(1, max_k + 1)

    for k in ks:
        Wk = compute_Wk(data, k)
        Wks.append(np.log(Wk))

        Wkbs_ref = []
        for _ in range(refs):
            random_data = np.random.uniform(np.min(data, axis=0), np.max(data, axis=0), size=shape)
            Wkb = compute_Wk(random_data, k)
            Wkbs_ref.append(np.log(Wkb))
        Wkbs.append(np.mean(Wkbs_ref))

        gap = np.mean(Wkbs_ref) - np.log(Wk)
        gaps.append(gap)

    return ks, gaps, Wks, Wkbs


def find_optimal_k(gaps):
    """Finds the optimal k using standard Gap Statistic heuristic."""
    for i in range(len(gaps)-1):
        if gaps[i] >= gaps[i+1] - np.std(gaps):
            return i + 1
    return len(gaps)


# === Main Usage ===
def give_res(imp):
    image_path = imp #"output_image.jpg"  # replace with your image
    #convert_tiff_to_jpg(image_path)
    pixels = load_image_pixels(image_path)

    ks, gaps, Wks, Wkbs = gap_statistic(pixels, refs=5, max_k=10)
    optimal_k = find_optimal_k(gaps)
    
    return optimal_k
#p_link="output_image.jpg"
#r=give_res(p_link)
#print("optimal number of cluster is : ",r)
# Plot
plt.figure(figsize=(8, 5))
plt.plot(ks, gaps, marker='o', color='b')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Gap Statistic")
plt.title(f"Optimal Clusters: {optimal_k}")
plt.grid(True)
plt.show()'''

'''import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, calinski_harabasz_score
from original_out import convert_tiff_to_jpg

# === Image Preprocessing ===
def load_image_pixels(image_path, grayscale=True):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(image_path, flag)
    if grayscale:
        pixels = image.reshape(-1, 1)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = image.reshape(-1, 3)
    return pixels

# === Gap Statistic ===
def compute_Wk(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    Wk = 0
    for i in range(n_clusters):
        cluster_data = data[labels == i]
        if cluster_data.shape[0] > 0:
            distances = pairwise_distances(cluster_data, [cluster_centers[i]])
            Wk += np.sum(distances ** 2)
    return Wk

def gap_statistic(data, refs=5, max_k=10):
    shape = data.shape
    gaps = []
    Wks = []
    Wkbs = []

    for k in range(1, max_k + 1):
        Wk = compute_Wk(data, k)
        Wks.append(np.log(Wk))

        Wkb_ref = []
        for _ in range(refs):
            random_data = np.random.uniform(np.min(data, axis=0), np.max(data, axis=0), size=shape)
            Wkb = compute_Wk(random_data, k)
            Wkb_ref.append(np.log(Wkb))

        gap = np.mean(Wkb_ref) - np.log(Wk)
        Wkbs.append(np.mean(Wkb_ref))
        gaps.append(gap)

    return list(range(1, max_k + 1)), gaps

def find_optimal_k_gap(gaps):
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - np.std(gaps):
            return i + 1
    return len(gaps)

# === Calinski-Harabasz Index ===
def find_optimal_k_ch(data, max_k=10):
    scores = []
    ks = range(2, max_k + 1)

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        labels = kmeans.labels_
        score = calinski_harabasz_score(data, labels)
        scores.append(score)

    optimal_k = ks[np.argmax(scores)]
    return optimal_k

# === Main Function ===
def get_optimal_clusters(image_path, max_k=10, pixel_limit=10000):
    # Convert TIFF if needed
    if image_path.lower().endswith((".tif", ".tiff")):
        image_path = convert_tiff_to_jpg(image_path)

    # Load grayscale pixels
    pixels = load_image_pixels(image_path, grayscale=True)
    num_pixels = pixels.shape[0]

    if num_pixels <= pixel_limit:
        # Use Gap Statistic
        print(f"[INFO] Using Gap Statistic for {num_pixels} pixels")
        ks, gaps = gap_statistic(pixels, refs=5, max_k=max_k)
        optimal_k = find_optimal_k_gap(gaps)
    else:
        # Use Calinski-Harabasz Index
        print(f"[INFO] Using Calinski-Harabasz Index for {num_pixels} pixels")
        optimal_k = find_optimal_k_ch(pixels, max_k=max_k)

    return optimal_k

# === Example Usage ===
def give_res(imp):
    image_path = imp #"shiv, manesh patil.png"
    optimal_k = get_optimal_clusters(image_path, max_k=10)
    if(optimal_k>5):
        optimal_k1=5
    else:
        optimal_k1=optimal_k
    #print("Optimal number of clusters:", optimal_k)
    return optimal_k,optimal_k1'''
    
'''if __name__ == "__main__":
    image_path = "shiv, manesh patil.png"
    optimal_k = get_optimal_clusters(image_path, max_k=10)
    print("Optimal number of clusters:", optimal_k)'''

import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from original_out import convert_tiff_to_jpg  # Your own TIFF-to-JPG converter

# === Load Image Pixels ===
def load_image_pixels(image_path, grayscale=True):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(image_path, flag)
    if grayscale:
        pixels = image.reshape(-1, 1)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = image.reshape(-1, 3)
    return pixels

# === Calinski-Harabasz Index Based Optimal K ===
def find_optimal_k_ch(data, max_k=10):
    scores = []
    ks = range(2, max_k + 1)

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        labels = kmeans.labels_
        score = calinski_harabasz_score(data, labels)
        scores.append(score)

    optimal_k = ks[np.argmax(scores)]
    return optimal_k

# === Main Function ===
def get_optimal_clusters(image_path, max_k=10):
    # Convert TIFF if needed
    if image_path.lower().endswith((".tif", ".tiff")):
        image_path = convert_tiff_to_jpg(image_path)

    # Load pixels (grayscale to save memory)
    pixels = load_image_pixels(image_path, grayscale=True)

    # Use CH Index
    print(f"[INFO] Using Calinski-Harabasz Index for {pixels.shape[0]} pixels")
    optimal_k = find_optimal_k_ch(pixels, max_k=max_k)

    return optimal_k

# === Wrapper Function ===
def give_res(image_path):
    optimal_k = get_optimal_clusters(image_path, max_k=10)
    optimal_k1 = min(optimal_k, 5)  # Cap at 5 if needed
    return optimal_k, optimal_k1

# === Example Usage ===
'''
if __name__ == "__main__":
    image_path = "shiv, manesh patil.png"
    opt_k, capped_k = give_res(image_path)
    print("Optimal number of clusters:", opt_k)
    print("Capped (<=5) clusters used for segmentation:", capped_k)
'''
