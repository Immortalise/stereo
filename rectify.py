# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import cv2
import numpy as np
import glob


# %%
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)

# objp 产生一个三维世界的点，因为是棋盘所以z轴为0
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints_l = [] # 3d point in real world space
objpoints_r = []
imgpoints_l = [] # 2d points in image plane.
imgpoints_r = [] # 2d points in image plane.
# print(objp)


# %%
images_l = sorted(glob.glob('/Users/triste/MarchingOn/project/stereo/left/*.jpg'))
print(len(images_l))
for fname in images_l:
    img_l = cv2.imread(fname)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_l, (7,6), None)
    if ret == True:
        objpoints_l.append(objp)
        imgpoints_l.append(corners)
    else:
        print(fname)
# print(len(imgpoints_l))
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints_l, imgpoints_l, gray_l.shape[::-1], None, None)


# %%
images_r = sorted(glob.glob('/Users/triste/MarchingOn/project/stereo/right/*.jpg'))
print(len(images_r))
for fname in images_r:
    # print(fname)
    img_r = cv2.imread(fname)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray_r, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints_r.append(objp)
        imgpoints_r.append(corners)
        # # 绘制查找到的角点
        # corners2 = cv2.cornerSubPix(gray_r, corners, (11,11), (-1,-1), criteria)
        # cv2.drawChessboardCorners(img_r, (7,6), corners2, ret)
        # cv2.imshow('img', img_r)
        # cv2.waitKey(-1)
    else:
        print(fname)
cv2.destroyAllWindows()
# print(len(imgpoints_r))
size = gray_r.shape[::-1]
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints_r, imgpoints_r, gray_r.shape[::-1], None, None)


# %%
# print(len(objpoints_l))
# print(len(imgpoints_l), len(imgpoints_r))


# %%


stereocalibration_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
rms, C1, dist1, C2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints_l,imgpoints_l,imgpoints_r,mtx_l,dist_l,mtx_r,dist_r,gray_r.shape[::-1],criteria = stereocalibration_criteria, flags = stereocalibration_flags)

#双目立体矫正及左右相机内参进一步修正
# print(T)
mean_error = 0
for i in range(len(objpoints_l)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

# %%
# #立体校正及深度图获取
# cv2.namedWindow("depth")
# def callbackFunc(e, x, y, f, p):
#     if e == cv2.EVENT_LBUTTONDOWN:
#         print(threeD[y][x])

# cv2.setMouseCallback("depth", callbackFunc, None)

R1,R2,P1,P2,Q,validPixROI1,validPixROI2 = cv2.stereoRectify(C1,dist1,C2,dist2,size,R,T)
print(P1)
print("\n=================")
print(P2)
print("\n=================")
print(Q)
left_map1, left_map2 = cv2.initUndistortRectifyMap(C1, dist1, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(C2, dist2, R2, P2, size, cv2.CV_16SC2)

frame1 = cv2.imread("/Users/triste/MarchingOn/project/stereo/left/left14.jpg")
frame2 = cv2.imread("/Users/triste/MarchingOn/project/stereo/right/right14.jpg")
# cv2.imshow("left", frame1)
# cv2.imshow("right", frame2)
img1_rectified = cv2.remap(frame1, left_map1, left_map2, cv2.INTER_LINEAR)
img2_rectified = cv2.remap(frame2, right_map1, right_map2, cv2.INTER_LINEAR)
# img1_rectified = img1_rectified*255
cv2.imshow("left", img1_rectified)
cv2.imshow("right", img2_rectified)
cv2.waitKey(-1)
imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
num = cv2.getTrackbarPos("num", "depth")
blockSize = cv2.getTrackbarPos("blockSize", "depth")
if blockSize % 2 == 0:
    blockSize += 1
if blockSize < 5:
    blockSize = 5
stereo = cv2.StereoBM_create(numDisparities=0, blockSize=5)
disparity = stereo.compute(imgL, imgR)
disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., Q) #此三维坐标点的基准坐标系为左侧相机坐标系
cv2.imshow("depth", disp)
cv2.waitKey(-1)
cv2.destroyAllWindows()

