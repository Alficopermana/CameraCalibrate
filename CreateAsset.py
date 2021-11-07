import numpy as np
import cv2 
import glob

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (7,7)
frameSize = (640,480)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

#size_of_chessboard_squares_mm = 20
#objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Camera using external (realtime)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True : 

    _, img = cap.read()
    # width = int(cap.get(3))
    # height = int(cap.get(4))

    img = np.array(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

    dst = None
    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria) 
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
        
        # ############## CALIBRATION #######################################################

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
        # print("Camera Calibrated : ", ret)
        print("\nCamera Matrix : \n", mtx)
        print("\nDistortion Parameters : \n", dist)
        print("\nRotation Vectors : \n", rvecs)
        print("\nTranslation Vectors : \n", tvecs)

        with open("ASSETS/CameraMatrix.txt", "w") as f:
            for i in mtx:
                f.write(str(i))
                print("\n")
        f.close()

        with open("ASSETS/Distortion.txt", "w") as f:
            for i in dist:
                f.write(str(i))
                print("\n")
        f.close()
                
    cv2.imshow('img', img)
    # cv2.imshow('dst', dst)

    if cv2.waitKey(1) == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()




# ############## CALIBRATION #######################################################

# ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# print("Camera Calibrated : ", ret)
# print("\nCamera Matrix : \n", cameraMatrix)
# print("\nDistortion Parameters : \n", dist)
# print("\nRotation Vectors : \n", rvecs)
# print("\nTranslation Vectors : \n", tvecs)

# ############## UNDISTORTION #####################################################

# img = cv.imread('cali5.png')
# h,  w = img.shape[:2]
# newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))



# # Undistort
# dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('caliResult1.png', dst)

# # Undistort with Remapping
# mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
# dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('caliResult2.png', dst)

# # Reprojection Error
# mean_error = 0

# for i in range(len(objpoints)):
#     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
#     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
#     mean_error += error

# print( "total error: {}".format(mean_error/len(objpoints)) )

