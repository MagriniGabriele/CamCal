import numpy as np
import cv2 as cv
import glob
import argparse
import os

def get_world_space_origin(cmtx, dist, img_path):

    frame = cv.imread(img_path, 1)

    #calibration pattern settings
    rows = 6
    columns = 8
    world_scaling = 1.0

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

    cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
    cv.putText(frame, "If you don't see detected points, try with a different image", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    cv.imshow('img', frame)
    cv.waitKey(0)

    ret, rvec, tvec = cv.solvePnP(objp, corners, cmtx, dist)
    R, _  = cv.Rodrigues(rvec) #rvec is Rotation matrix in Rodrigues vector form

    return R, tvec

def get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0, 
                                 cmtx1, dist1, R_01, T_01,
                                 image_path0,
                                 image_path1):

    frame0 = cv.imread(image_path0, 1)
    frame1 = cv.imread(image_path1, 1)

    unitv_points = 5 * np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
    #axes colors are RGB format to indicate XYZ axes.
    colors = [(0,0,255), (0,255,0), (255,0,0)]

    #project origin points to frame 0
    points, _ = cv.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame0, origin, _p, col, 2)

    #project origin points to frame1
    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01
    points, _ = cv.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame1, origin, _p, col, 2)

    cv.imshow('frame0', frame0)
    cv.imshow('frame1', frame1)
    cv.waitKey(0)

    return R_W1, T_W1


#open paired calibration frames and stereo calibrate for cam0 to cam1 coorindate transformations
def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1):
    #read the synched frames
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))

    #open images
    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]

    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    #calibration pattern settings
    rows = 6
    columns = 8
    world_scaling = 1.

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space

    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:

            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0,0].astype(np.int32)
            p0_c2 = corners2[0,0].astype(np.int32)

            cv.putText(frame0, 'O', (p0_c1[0], p0_c1[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame0, (rows,columns), corners1, c_ret1)
            cv.imshow('img', frame0)

            cv.putText(frame1, 'O', (p0_c2[0], p0_c2[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame1, (rows,columns), corners2, c_ret2)
            cv.imshow('img2', frame1)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx0, dist0,
                                                                 mtx1, dist1, (width, height), criteria = criteria, flags = stereocalibration_flags)

    print('rmse: ', ret)
    cv.destroyAllWindows()
    return R, T



#Calibrate single camera to obtain camera intrinsic parameters from saved frames.
def calibrate_camera_for_intrinsic_parameters(images_prefix):
    
    images_names = glob.glob(images_prefix)

    #read all frames
    images = [cv.imread(imname, 1) for imname in images_names]

    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard. 
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = 6
    columns = 8
    world_scaling = 1.

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space


    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:

            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv.putText(frame, 'If detected points are poor, press "s" to skip this sample', (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

            cv.imshow('img', frame)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints.append(corners)


    cv.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', cmtx)
    print('distortion coeffs:', dist)

    return cmtx, dist

######### OLD VERSION #########
# class StereoCalibration(object):
#     def __init__(self, filepath):
#         # termination criteria
#         self.criteria = (cv2.TERM_CRITERIA_EPS +
#                          cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#         self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
#                              cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

#         # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#         self.objp = np.zeros((8*6, 3), np.float32)
#         self.objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

#         # Arrays to store object points and image points from all the images.
#         self.objpoints = []  # 3d point in real world space
#         self.imgpoints_l = []  # 2d points in image plane.
#         self.imgpoints_r = []  # 2d points in image plane.

#         self.cal_path = filepath
#         self.read_images(self.cal_path)

#     def read_images(self, cal_path):
#         images_right = glob.glob(cal_path + 'RIGHT/*.png')
#         images_left = glob.glob(cal_path + 'LEFT/*.png')
#         # Sort the images names by number
#         images_right.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
#         images_left.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))



#         for i, fname in enumerate(images_right[:14]):
#             img_l = cv2.imread(images_left[i])
#             img_r = cv2.imread(images_right[i])

#             gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
#             gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

#             # Find the chess board corners
#             ret_l, corners_l = cv2.findChessboardCorners(gray_l, (8, 6), None)
#             ret_r, corners_r = cv2.findChessboardCorners(gray_r, (8, 6), None)

#             # If found, add object points, image points (after refining them)
#             self.objpoints.append(self.objp)

#             if ret_l is True:
#                 rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
#                                       (-1, -1), self.criteria)
#                 self.imgpoints_l.append(corners_l)

#                 # Draw and display the corners
#                 ret_l = cv2.drawChessboardCorners(img_l, (8, 6),
#                                                   corners_l, ret_l)
#                 cv2.imshow(images_left[i], img_l)
#                 cv2.waitKey(200)

#             if ret_r is True:
#                 rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
#                                       (-1, -1), self.criteria)
#                 self.imgpoints_r.append(corners_r)

#                 # Draw and display the corners
#                 ret_r = cv2.drawChessboardCorners(img_r, (8, 6),
#                                                   corners_r, ret_r)
#                 cv2.imshow(images_right[i], img_r)
#                 cv2.waitKey(200)
#             img_shape = gray_l.shape[::-1]

#         rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
#             self.objpoints, self.imgpoints_l, img_shape, None, None)
#         rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
#             self.objpoints, self.imgpoints_r, img_shape, None, None)

#         self.camera_model = self.stereo_calibrate(img_shape)

#     def stereo_calibrate(self, dims):
#         flags = 0
#         flags |= cv2.CALIB_FIX_INTRINSIC
#         # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
#         flags |= cv2.CALIB_USE_INTRINSIC_GUESS
#         flags |= cv2.CALIB_FIX_FOCAL_LENGTH
#         # flags |= cv2.CALIB_FIX_ASPECT_RATIO
#         flags |= cv2.CALIB_ZERO_TANGENT_DIST
#         # flags |= cv2.CALIB_RATIONAL_MODEL
#         # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
#         # flags |= cv2.CALIB_FIX_K3
#         # flags |= cv2.CALIB_FIX_K4
#         # flags |= cv2.CALIB_FIX_K5

#         stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
#                                 cv2.TERM_CRITERIA_EPS, 100, 1e-5)
#         ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
#             self.objpoints, self.imgpoints_l,
#             self.imgpoints_r, self.M1, self.d1, self.M2,
#             self.d2, dims,
#             criteria=stereocalib_criteria, flags=flags)

#         print('Intrinsic_mtx_1', M1)
#         print('dist_1', d1)
#         print('Intrinsic_mtx_2', M2)
#         print('dist_2', d2)
#         print('R', R)
#         print('T', T)
#         print('E', E)
#         print('F', F)
#         print('rmse', ret)

#         # for i in range(len(self.r1)):
#         #     print("--- pose[", i+1, "] ---")
#         #     self.ext1, _ = cv2.Rodrigues(self.r1[i])
#         #     self.ext2, _ = cv2.Rodrigues(self.r2[i])
#         #     print('Ext1', self.ext1)
#         #     print('Ext2', self.ext2)

#         print('')

#         camera_model = dict([('M1', M1), ('M2', M2), ('d1', d1),
#                             ('d2', d2), ('rvecs1', self.r1),
#                             ('rvecs2', self.r2), ('R', R), ('T', T),
#                             ('E', E), ('F', F)])

#         cv2.destroyAllWindows()
#         return camera_model
    

# def main(cal_path="Calibration/"):
#     cal = StereoCalibration(cal_path)
#     print("Stereo calibration done")
#     print("Camera model", cal.camera_model)

#     cv2.destroyAllWindows()
#     return cal.camera_model

# if __name__ == '__main__':
#     main("Calibration/")
    

   
