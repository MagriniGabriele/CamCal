
import numpy as np
import cv2

# Function to detect chessboard corners in a single image
def detect_corners(image, board_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners

# Function to detect chessboard corners in multiple pairs of images
def detect_corners_multiple(images, board_size):
    objpoints = []  # 3D point in real world space
    imgpoints = []  # 2D points in image plane

    for img_pair in images:
        ret1, corners1 = detect_corners(img_pair[0], board_size)
        ret2, corners2 = detect_corners(img_pair[1], board_size)

        if ret1 and ret2:
            objp = np.zeros((np.prod(board_size), 3), dtype=np.float32)
            objp[:, :2] = np.indices(board_size).T.reshape(-1, 2)

            objpoints.append(objp)
            imgpoints.append([corners1, corners2])

    return objpoints, imgpoints

# Load images
image_pairs = [
    (cv2.imread('12_event.png'), cv2.imread('12_rgb.png')),
    (cv2.imread('13_event.png'), cv2.imread('13_rgb.png')),
    (cv2.imread('10_event.png'), cv2.imread('10_rgb.png')),
    (cv2.imread('11_event.png'), cv2.imread('13_rgb.png')),
    # Add more pairs as needed
]
board_size = (8, 6)  # Adjust according to the chessboard size

# Detect chessboard corners in multiple pairs of images
objpoints, imgpoints = detect_corners_multiple(image_pairs, board_size)

# Calibrate camera using multiple pairs of images
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, [pair[0] for pair in imgpoints], image_pairs[0][0].shape[:2], None, None)

# Compute homography using each pair of images
homographies = []
for corners_pair in imgpoints:
    H, _ = cv2.findHomography(corners_pair[0], corners_pair[1], cv2.RANSAC)
    homographies.append(H)

# Average homography from all pairs
avg_H = np.mean(homographies, axis=0)

print("Average Homography matrix:")
print(avg_H)

H = avg_H

# Test also on another pair of images
image1 = cv2.imread('14_event.png')
image2 = cv2.imread('14_rgb.png')
# Detect chessboard corners in both images
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
board_size = (8, 6)  # Adjust according to the chessboard size

# Find corners in image1
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
ret1, corners1 = cv2.findChessboardCorners(gray_image1, board_size, None)

if ret1:
    # Compute homography from image1 to image2
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    ret2, corners2 = cv2.findChessboardCorners(gray_image2, board_size, None)
    if ret2:
        # Project corners from image1 to image2 using homography
        corners1_reshaped = corners1.reshape(-1, 1,2)
        projected_corners = cv2.perspectiveTransform(corners1_reshaped, H)
        

        # Draw circles on image2 at projected corners from image1S
        for corner in projected_corners:
            print(corner)
            x, y = corner[0][0], corner[0][1]
            
            cv2.circle(image2, tuple(corner.squeeze().astype(int)), 5, (0, 255, 0), -1)

        # Draw also the corners from image 2
        cv2.drawChessboardCorners(image2, board_size, corners2, ret2)


        # Display results
        cv2.imshow('Image 1', image1)
        cv2.imshow('Image 2 with projected corners', image2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Chessboard corners not found in image2.")