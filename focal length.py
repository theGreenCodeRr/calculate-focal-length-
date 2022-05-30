import cv2
import numpy as np


img = cv2.imread('chessboard18.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# prepare object points
obj_grid = np.zeros((6 * 9, 3), np.float32)
obj_grid[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object  and image points
objpoints = []
imgpoints = []

# find chessboard corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # termination criteria
retval, corners = cv2.findChessboardCorners(gray, (9, 6), None)
if retval:
    objpoints.append(obj_grid)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)
    # plot corners
    img = cv2.drawChessboardCorners(img, (7, 6), corners2, retval)
    cv2.imshow('img', img)
    cv2.waitKey(2000)

# calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def undistort(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    # cv2.imshow('img', img)
    # cv2.imshow('Result.png', dst)
    # cv2.waitKey(5000)

    # plt.figure(figsize =(10, 5))
    # plt.subplot(121)
    # plt.imshow(img)
    # plt.title('Original')
    # plt.subplot(122)
    # plt.imshow(dst)
    # plt.title('Corrected')
    # plt.show()


# print("Camera matrix: \n", mtx)
# Calculate Focal Length in mm F = (fx in pixels/img width in pixels) * Sensor Width (22.3)
focal_length = int(mtx[0, 0] / img.shape[1] * 22.3)
print('\nFocal Length:', focal_length, "mm")
