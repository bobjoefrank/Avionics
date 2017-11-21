import cv2
import numpy as np
from matplotlib import pyplot as plt


def nothing(x):
    pass

def _create_sized_window(name: str, image) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.resizeWindow(name, 1200, 800)


def _display_canny_edge(image, lower_bound = 100, upper_bound = 200):

    edges = cv2.Canny(image, lower_bound, upper_bound)

    _create_sized_window('', edges)


def _get_surf_knn_matched(query_image, train_image):
    """Returns the final frame after mathcing surf descriptors with knn bf matcher"""

    surf = cv2.xfeatures2d_SURF.create(400)

    # Get keypoints and Descriptors of both images
    kp_query, des_query = surf.detectAndCompute(query_image, None)
    kp_train, des_train = surf.detectAndCompute(train_image, None)

    # Create matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_query, des_train, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    return cv2.drawMatchesKnn(query_image, kp_query, train_image, kp_train, good, None, flags=2)


def _get_surf_bf_matched(query_image, train_image):
    """Returns the final frame after matching surf descriptors with a brute force matcher"""

    surf = cv2.xfeatures2d_SURF.create(400)

    # Get keypoints and Descriptors of both images
    kp_query, des_query = surf.detectAndCompute(query_image, None)
    kp_train, des_train = surf.detectAndCompute(train_image, None)

    # Create matcher
    bf = cv2.BFMatcher()
    matches = bf.match(des_query, des_train)

    matches = sorted(matches, key = lambda x:x.distance)

    return cv2.drawMatches(query_image, kp_query, train_image, kp_train, matches[:100], None, flags=2)


def _get_surf_flann_matched(query_image, train_image):
    """Returns the final frame after matching surf descriptors with a FLANN matcher"""

    surf = cv2.xfeatures2d_SURF.create(400)

    # Get keypoints and Descriptors of both images
    kp_query, des_query = surf.detectAndCompute(query_image, None)
    kp_train, des_train = surf.detectAndCompute(train_image, None)

    # FLANN params
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_query, des_train, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)



    return cv2.drawMatchesKnn(query_image, kp_query, train_image, kp_train, matches, None, **draw_params)


def _get_hsv_trackbar_pos():
    """Returns a tuple of lists containing the values of the trackbar"""

    min_h = cv2.getTrackbarPos('min_h', 'image')
    min_s = cv2.getTrackbarPos('min_s', 'image')
    min_v = cv2.getTrackbarPos('min_v', 'image')

    max_h = cv2.getTrackbarPos('max_h', 'image')
    max_s = cv2.getTrackbarPos('max_s', 'image')
    max_v = cv2.getTrackbarPos('max_v', 'image')

    return ([min_h, min_s, min_v], [max_h, max_s, max_v])

queryImage = cv2.imread('test_area1.png')
# trainImage = cv2.imread('single_target.png')

# queryImage = cv2.GaussianBlur(queryImage, (5,5), 0)

hsvQuery = cv2.cvtColor(queryImage, cv2.COLOR_BGR2HSV)
# hsvTrain = cv2.cvtColor(trainImage, cv2.COLOR_BGR2HSV)
#
# query_h, query_s, query_v = cv2.split(hsvQuery)
# train_h, train_s, train_v = cv2.split(hsvTrain)

surf = cv2.xfeatures2d_SURF.create(400)

keypoints, descriptors = surf.detectAndCompute(hsvQuery, None)

final = cv2.drawKeypoints(hsvQuery, keypoints,None)
display = cv2.drawKeypoints(hsvQuery, keypoints, None)


kp_point = keypoints[0]

print(kp_point.pt)

keyimage = final[int(kp_point.pt[1]-50):int(kp_point.pt[1]+50),int(kp_point.pt[0]-50):int(kp_point.pt[0]+50)]


# Threshold the color values to remove green from image
# or
# Create a shape outline and attempt to feature match the shape with a thresholded image

# Template matcher?

_create_sized_window('final', display)
_create_sized_window('roi', keyimage)
cv2.waitKey(0)


# TRACKBAR
# # Create a black image, a window
# img = np.zeros((300,512,3), np.uint8)
# cv2.namedWindow('image')
# # create trackbars for color change
# cv2.createTrackbar('min_h','image',0,255,nothing)
# cv2.createTrackbar('min_s','image',0,255,nothing)
# cv2.createTrackbar('min_v','image',0,255,nothing)
#
# cv2.createTrackbar('max_h','image',0,255,nothing)
# cv2.createTrackbar('max_s','image',0,255,nothing)
# cv2.createTrackbar('max_v','image',0,255,nothing)

# while not cv2.waitKey(1) & 0xFF == 27:
#
#     lower_values, upper_values = _get_hsv_trackbar_pos()
#     lower = np.array(lower_values)
#     upper = np.array(upper_values)
#
#     thresholded_query = cv2.inRange(hsvQuery, lower, upper)
#
#     _create_sized_window('similarities', thresholded_query)

cv2.destroyAllWindows()