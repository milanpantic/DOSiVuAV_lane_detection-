import numpy as np
import cv2
import glob

def calibrateCamera(calibrationDir, rows = 0, cols = 0):

    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
    
    objectPoints = np.zeros((rows * cols, 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    objectPointsArray = []
    imgPointsArray = []

    calibrationImgPath = calibrationDir + "calibration*.jpg"

    for path in glob.glob(calibrationImgPath):
        print(calibrationImgPath)
        
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objectPointsArray.append(objectPoints)
            imgPointsArray.append(corners)

            cv2.drawChessboardCorners(img, (rows, cols), corners, ret)
        
        # Display the image
        # cv2.imshow('chess board', img)
        # cv2.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)
    np.savez('camera_cal/calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    error = 0

    for i in range(len(objectPointsArray)):
        imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
        error += cv2.norm(imgPointsArray[i], imgPoints, cv2.NORM_L2) / len(imgPoints)

    print("Total error: ", error / len(objectPointsArray))

def drawCurve(image, points, color):
    """
    Fits a second-order curve through a set of points and draws it on the image.
    """
    if len(points) < 3:
        return image 

    xCoords = np.array([p[0] for p in points])
    yCoords = np.array([p[1] for p in points])

    fit = np.polyfit(xCoords, yCoords, 2)
    a, b, c = fit

    height, width = image.shape[:2]
    curvePoints = []
    for x in range(min(xCoords), max(xCoords) + 1):
        y = int(a * x**2 + b * x + c)
        if 0 <= y < height: 
            curvePoints.append((x, y))

    curvePoints = np.array([curvePoints], dtype=np.int32)
    cv2.polylines(image, curvePoints, isClosed=False, color=color, thickness=30)

    return image

def main():

    calibrateCamera("camera_cal/", 6, 9)

    calibratio = np.load('camera_cal/calib.npz')
    mtx = calibratio['mtx']
    dist = calibratio['dist']
    rvecs = calibratio['rvecs']
    tvecs = calibratio['tvecs']

    ##########################################
    ####              Tasks               ####
    ##########################################
    ###          Camera Calibration       ####

    originalChessImage = cv2.imread('camera_cal/calibration3.jpg')
    chessH, chessW = originalChessImage.shape[:2]
    newCameraMtxChess, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (chessW, chessH), 1, (chessW, chessH))
    undistortedChessImg = cv2.undistort(originalChessImage, mtx, dist, None, newCameraMtxChess)
    
    cv2.imshow('undistortedChessImg', np.hstack((originalChessImage, undistortedChessImg)))

    cap = cv2.VideoCapture('test_videos/project_video03.mp4')
    allLinePoints = []

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Video no longer open")
            break   

        h, w = img.shape[:2]

        newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)
        cv2.imshow('undistorted orignal image', np.hstack((img, undistortedImg)))


        srcPoints = np.float32([
            [0, h],   # Bottom-left corner
            [w, h], # Bottom-right corner
            [w*61/100, h*64/100], # Top-right corner
            [w*46/100, h*64/100]  # Top-left corner
        ])

        dstPoints = np.float32([
            [0, h],   # Bottom-left corner
            [w, h], # Bottom-right corner
            [w, 0],   # Top-right corner
            [0, 0]      # Top-left corner
        ])

        mask = np.ones((h, w), dtype=np.uint8) * 255

        srcPoints_int = srcPoints.astype(np.int32)

        cv2.fillPoly(mask, [srcPoints_int], (0, 0, 0))

        maskedImage = cv2.bitwise_and(undistortedImg, undistortedImg, mask=mask)

        transformMatrix = cv2.getPerspectiveTransform(srcPoints, dstPoints)
        warpedImage = cv2.warpPerspective(undistortedImg, transformMatrix, (w, h))

        warpedImage2 = warpedImage.copy()

        cv2.imshow('warpedImage', np.hstack((undistortedImg, warpedImage)))
        
        mask = np.ones((h, w), dtype=np.uint8) * 255

        srcPoints_int = srcPoints.astype(np.int32)

        cv2.fillPoly(mask, [srcPoints_int], (0, 0, 0))

        midPoint = w // 2
        rightLane = list()
        leftLane = list()

        gray = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray, (7, 7), 0)
        ret, thrash = cv2.threshold(gaussian, 135, 150, cv2.THRESH_BINARY)

        canny = cv2.Canny(thrash, 130, 150)
    
        cv2.imshow("Binary Image", canny)

        lines = cv2.HoughLinesP(canny, 1, np.pi / 180, threshold=25, minLineLength=200, maxLineGap=200)
        if lines is not None:
            for points in lines:
                x1, y1, x2, y2 = np.array(points[0])

                x1, y1, x2, y2 = points[0]
                p1, p2 = (x1, y1), (x2, y2)
                
                allLinePoints.append(p1)
                allLinePoints.append(p2)

                cv2.line(warpedImage2, p1, p2, 255)

                if (x1 > midPoint):
                    rightLane.append(p1)
                    rightLane.append(p2)
                    
                else:
                    leftLane.append(p1)
                    leftLane.append(p2)

        rightLineDrawn = drawCurve(warpedImage,rightLane, 255)
        leftLineDrawn = drawCurve(warpedImage,leftLane, 255)

        inverseMatrix = cv2.getPerspectiveTransform(dstPoints, srcPoints)

        reconstructedImage = cv2.warpPerspective(warpedImage, inverseMatrix, (w, h))

        finalImage = cv2.bitwise_or(maskedImage, reconstructedImage)

        cv2.imshow('finalImage done', np.hstack((undistortedImg, finalImage)))

        if cv2.waitKey(100) & 0xFF == 27: break 
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()