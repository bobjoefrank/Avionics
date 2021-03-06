import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2
import numpy as np

# Get better values of these from testing with images we get from our UAV
_SURF_THRESHOLD = 1500
_KEYPOINT_GROUPING_DISTANCE = 100
_HALFSIZE_OF_ROI_BOX = 50



def _create_sized_window(name: str, image, size_x: int, size_y: int) -> None:
    """Creates a window of 1200 by 800 for the image"""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.resizeWindow(name, size_x, size_y)


def points_are_within(distance: int, point1: cv2.KeyPoint, point2: cv2.KeyPoint) -> bool:

    if abs(point1.pt[0] - point2.pt[0]) <= distance and abs(point1.pt[1] - point2.pt[1]) < distance:
        return True
    else:
        return False


def _group_keypoints(kp: [cv2.KeyPoint]) -> list:
    """Returns the 'strongest' keypoint out of groups of keypoints that are in close proximity to one another"""

    current_index = 0

    # Compare all points in the list to all points in the list. If the points are within distance, check the response
    # strength. If the point_b is greater we set point_a = point_b otherwise dont do anything since point_a will already
    # be larger
    for point_a in kp:
        for point_b in kp:

            if points_are_within(_KEYPOINT_GROUPING_DISTANCE, point_a, point_b):
                if point_b.response > point_a.response:
                    kp[current_index] = point_b

        current_index += 1

    # Convert to set to remove duplicates
    kp = set(kp)

    # Return as a list
    return list(kp)


def _get_list_of_roi(image, kp: [cv2.KeyPoint]) -> list:
    """Takes the keypoints and returns a list of images that are regions around the keypoints"""

    roi_list = []

    height, width = image.shape[:2]

    for point in kp:
        x = []
        y = []

        x.append(int(point.pt[0] - _HALFSIZE_OF_ROI_BOX))
        x.append(int(point.pt[0] + _HALFSIZE_OF_ROI_BOX))
        y.append(int(point.pt[1] - _HALFSIZE_OF_ROI_BOX))
        y.append(int(point.pt[1] + _HALFSIZE_OF_ROI_BOX))

        # If any of the values are less than 0 -> set them to 0, if any values > width or height -> set as width/height
        x = [0 if x_value < 0 else width - 1 if x_value > width else x_value for x_value in x]
        y = [0 if y_value < 0 else height - 1 if y_value > height else y_value for y_value in y]

        roi_list.append(image[y[0]:y[1], x[0]:x[1]])

    return roi_list


def run_object_detection():
    """Runs the object detection module"""

    # Get image and convert to HSV
    image_rgb = cv2.imread('../pictures/test_area1.png')
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)

    height, width = image_rgb.shape[:2]
    _create_sized_window('lll', image_rgb, width, height)
    # Create our SURF object and run the algorithm on our image
    surf = cv2.xfeatures2d.SURF_create(_SURF_THRESHOLD)
    keypoints = surf.detect(image_hsv, None)

    # Pass in a copy of the keypoints list, otherwise it passes by reference and not by value
    grouped_kp = _group_keypoints(keypoints[:])

    # For debugging
    print(len(keypoints))
    print(len(grouped_kp))

    grouped_kp_image = cv2.drawKeypoints(image_hsv, grouped_kp, None, (255, 0, 0), 4)
    # ungrouped_kp_image = cv2.drawKeypoints(image_hsv, keypoints, None, (255, 0, 0), 4)

    # Get the ROI's from the keypoints and for each ROI display it
    region_of_interest = _get_list_of_roi(grouped_kp_image, grouped_kp)
    for img in region_of_interest:
        _create_sized_window(f'{img}', img, 300, 300)


    # _create_sized_window('GROUPED', grouped_kp_image, 1200, 800)
    # _create_sized_window('NOT GROUPED', ungrouped_kp_image, 1200, 800)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_object_detection()
