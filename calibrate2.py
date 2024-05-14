import numpy as np
import cv2
import glob

# Define chessboard parameters
chessboard_size = (8, 6)  # Inner corners of the chessboard
square_size = 0.0254  # Size of one square in meters

# Termination criteria for corner detection
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all the images.
objpoints_single = []  # 3d point in real world space for single calibration
imgpoints1_single = []  # 2d points in image plane for camera 1 for single calibration
imgpoints2_single = []  # 2d points in image plane for camera 2 for single calibration

objpoints_stereo = []  # 3d point in real world space for stereo calibration
imgpoints1_stereo = []  # 2d points in image plane for camera 1 for stereo calibration
imgpoints2_stereo = []  # 2d points in image plane for camera 2 for stereo calibration

# Load images for single calibration from both cameras
images1_single = glob.glob('Calibrate_Alternative/LEFT_SINGLE/*.png')
images2_single = glob.glob('Calibrate_Alternative/RIGHT_SINGLE/*.png')

print(images1_single)
print(images2_single)

for img_path1, img_path2 in zip(images1_single, images2_single):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, chessboard_size, None)

    if ret1 and ret2:
        objpoints_single.append(objp)

        # Refine corner locations
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        imgpoints1_single.append(corners1)

        corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
        imgpoints2_single.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img1, chessboard_size, corners1, ret1)
        cv2.imshow('Chessboard Corners - Single Calibration (Camera 1)', img1)
        cv2.drawChessboardCorners(img2, chessboard_size, corners2, ret2)
        cv2.imshow('Chessboard Corners - Single Calibration (Camera 2)', img2)
        cv2.waitKey(500)

# Load images for stereo calibration from both cameras
images1_stereo = glob.glob('Calibrate_Alternative/LEFT/*.png')
images2_stereo = glob.glob('Calibrate_Alternative/RIGHT/*.png')

for img_path1, img_path2 in zip(images1_stereo, images2_stereo):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, chessboard_size, None)

    if ret1 and ret2:
        objpoints_stereo.append(objp)

        # Refine corner locations
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        imgpoints1_stereo.append(corners1)

        corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
        imgpoints2_stereo.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img1, chessboard_size, corners1, ret1)
        cv2.imshow('Chessboard Corners - Stereo Calibration (Camera 1)', img1)
        cv2.drawChessboardCorners(img2, chessboard_size, corners2, ret2)
        cv2.imshow('Chessboard Corners - Stereo Calibration (Camera 2)', img2)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Single camera calibration
ret, mtx1, dist1, _, _ = cv2.calibrateCamera(objpoints_single, imgpoints1_single, gray1.shape[::-1], None, None)
ret, mtx2, dist2, _, _ = cv2.calibrateCamera(objpoints_single, imgpoints2_single, gray2.shape[::-1], None, None)

# Stereo calibration
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(objpoints_stereo, imgpoints1_stereo, imgpoints2_stereo, mtx1, dist1, mtx2, dist2, gray1.shape[::-1], criteria=criteria, flags=flags)

# Rectification
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, gray1.shape[::-1], R, T)

# Compute the rectification maps for mapping the pixel coordinates
map1x, map1y = cv2.initUndistortRectifyMap(mtx1, dist1, R1, P1, gray1.shape[::-1], cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(mtx2, dist2, R2, P2, gray2.shape[::-1], cv2.CV_32FC1)

# Remap the chessboard corners from camera 1 to camera 2
imgpoints1_stereo = np.array(imgpoints1_stereo)
remapped_corners = cv2.convertPointsToHomogeneous(imgpoints1_stereo).reshape(-1, 3, 1)
remapped_corners = cv2.perspectiveTransform(remapped_corners, Q)


# Print mapped corners
print("Mapped corners in Camera 2:")
for point in remapped_corners:
    print(point.squeeze())

# Take an image used for stereo calibration from camera 1
test_img_path1 = 'Calibrate_Alternative/LEFT/14_rgb.png'
test_img1 = cv2.imread(test_img_path1)
gray_test_img1 = cv2.cvtColor(test_img1, cv2.COLOR_BGR2GRAY)

# Take an image used for stereo calibration from camera 2
test_img_path2 = 'Calibrate_Alternative/RIGHT/14_event.png'
test_img2 = cv2.imread(test_img_path2)

# Find chessboard corners in the test image from camera 1
ret_test1, corners_test1 = cv2.findChessboardCorners(gray_test_img1, chessboard_size, None)
if ret_test1:
    corners_test_refined1 = cv2.cornerSubPix(gray_test_img1, corners_test1, (11, 11), (-1, -1), criteria)

    # Map the chessboard corners from camera 1 to camera 2
    remapped_corners_test1 = cv2.convertPointsToHomogeneous(corners_test_refined1).reshape(-1, 3, 1)
    remapped_corners_test1 = cv2.perspectiveTransform(remapped_corners_test1, Q)

    # Draw corners on the test image from camera 1
    img_with_corners1 = cv2.drawChessboardCorners(test_img1, chessboard_size, corners_test_refined1, ret_test1)

    # Draw mapped points on the corresponding image from camera 2
    img_with_mapped_points2 = np.copy(test_img2)
    for point in remapped_corners_test1:
        point_2d = tuple(point.squeeze().astype(int))
        cv2.circle(img_with_mapped_points2, point_2d, 5, (0, 255, 0), -1)

    # Display the test image from camera 1 with chessboard corners and the test image from camera 2 with mapped points
    cv2.imshow('Test Image from Camera 1 with Corners', img_with_corners1)
    cv2.imshow('Test Image from Camera 2 with Mapped Points', img_with_mapped_points2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Chessboard corners not found in the test image from camera 1.")
