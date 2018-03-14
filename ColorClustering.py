from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import numpy as np

def __centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def __plot_colors(hist, centroid):
    # initialize the bar chart representing the relative frequency
	# of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
	# each cluster
    max_percent = 0
    max_color = None
    for(percent, color) in zip(hist, centroid):
        if percent > max_percent:
            max_percent = percent
            max_color = color
    return max_color

def LineDetection(original):
    dst = cv2.Canny(original, 50, 200)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLinesP(dst, 1, np.pi/180.0, 40, np.array([]), 50, 10)
    a,b,c = lines.shape
    print(lines.shape)
    if(a > 50):
        return True
    else:
        return False
    

def ColorDetection(original):

    image = original.reshape((original.shape[0] * original.shape[1], 3))
    clt = KMeans(n_clusters = 3)
    clt.fit(image)
    hist = __centroid_histogram(clt)
    colors_stats = __plot_colors(hist, clt.cluster_centers_)
    __line_detection(original)
    return colors_stats


