import cv2
import numpy as np
import glob
import math
import modern_robotics as mr
import win32gui
from objloader_simple import *
import pygame
from threading import Thread
import socket
import time
import random as random
from tkinter import *
from PIL import ImageTk
from PIL import Image as PilImage
from mstrio import microstrategy
import xbox360_controller
pygame.init()

ANIMAL_STATS_CUBE_ID = '13D00F8811E9328F86670080EFA5E658'
ANIMAL_WORDS_CUBE_ID = 'A83DFB8611E932A783C60080EF554555'

conn = microstrategy.Connection(base_url="https://env-132697.customer.cloud.mic\
rostrategy.com/MicroStrategyLibrary/api", username="hackathon", \
password="m$trhackathon2019", project_name="MicroStrategy Tutorial")
conn.connect()
#This code detects if a button state is changed and prints it to the terminal
#screen = pygame.display.set_mode(size)

#Variable Declarations ===============================================================
controller = xbox360_controller.Controller(0)
animal = OBJ("bear.obj", swapyz=False)

def UI():
    def get_query(query):
        global consta
        # Check for empty String
        if len(query) == 0:
            return False

        conn.connect()

        # Get first cube
        asc_df = conn.get_cube(cube_id=ANIMAL_STATS_CUBE_ID)
        # Check if any results for query
        res = asc_df.loc[asc_df['Animal'] == query.title()]

        if res.empty:
            print("No results for that query")
            return False
        else:
            diet = res.iloc[0]['Diet']
            lifespan = res.iloc[0]['Lifespan']
            size = res.iloc[0]['Size']
            consta = res.iloc[0]['Conservation Status']
            details_text.set(res.iloc[0]['Details'])
            top_speed = res.iloc[0]['Top Speed']
            weight = res.iloc[0]['Weight']
            habitat_text.set(res.iloc[0]['Habitat'])
            return True

    def search_data():
        global consta
        global animal
        query = searchEntry.get()
        animal = setAniString(query)
        print('Sending query: %s' % query)
        if not get_query(query):
            searchEntry.insert(0, "No results for the query: ")
            return
        # Populate fields
        name.set(query.title())
        frame.grid()
        conservation.grid()
        # Set conservation
        status_EX.grid_remove()
        status_EW.grid_remove()
        status_CR.grid_remove()
        status_EN.grid_remove()
        status_VU.grid_remove()
        status_NT.grid_remove()
        status_LC.grid_remove()
        print(consta)
        if 'extinct' in consta.lower():
            status_EX.grid()
        elif 'extinct in wild' in consta.lower():
            status_EW.grid()
        elif 'critically endangered' in consta.lower():
            status_CR.grid()
        elif 'endangered' in consta.lower():
            status_EN.grid()
        elif 'threatened' or 'vulnerable' in consta.lower():
            status_VU.grid()
        elif 'near threatened' in consta.lower():
            status_NT.grid()
        elif 'least concerned' in consta.lower():
            status_LC.grid()

    master = Tk(screenName='BestiARy')
    master.title('BestiARy')
    path = 'Title.png'
    img = ImageTk.PhotoImage(PilImage.open(path))
    title = Label(master, image=img).grid(row=0, column=0, columnspan=2)

    master.rowconfigure(0, weight=1)
    master.columnconfigure(0, weight=1)

    searchEntry = Entry(master, width=50)
    searchEntry.grid(row=1, column=0)
    Button(master, text='Search', command=search_data).grid(row=1, column=1)

    frame = Frame(master)
    frame.grid(row=2, column=0, columnspan=2)
    name = StringVar('')
    common_name = Label(frame, textvariable=name, font=("Helvetica", 18)).grid(row=0, column=1)
    habitat_text = StringVar('')
    habitatLabel = Message(frame, text="Habitat: ", width=100, font=("Helvetica", 14))
    habitatLabel.grid(row=1, column=0)
    habitatText = Message(frame, textvariable=habitat_text, width=650)
    habitatText.grid(row=1, column=1, columnspan=2)
    details_text = StringVar('')
    details = Message(frame, textvariable=details_text, width=650)
    details.grid(row=2, column=0, columnspan=3)
    details.grid_remove()
    frame.grid_remove()

    consta = ''
    conservation = Frame(master)
    conservation_status = Message(conservation, width=650, text='Conservation Status', font=("Helvetica", 12))
    conservation_status.grid(row=0, column=0, columnspan=7)
    conservation.grid(row=3, column=0, columnspan=2)
    status_EX = Message(conservation, text='EX', bg='white', font=("Helvetica", 16))
    status_EX.grid(row=1, column=0)
    status_EX.grid_remove()
    status_EW = Message(conservation, text='EW', bg='gray', font=("Helvetica", 16))
    status_EW.grid(row=1, column=1)
    status_EW.grid_remove()
    status_CR = Message(conservation, text='CR', bg='red', font=("Helvetica", 16))
    status_CR.grid(row=1, column=2)
    status_CR.grid_remove()
    status_EN = Message(conservation, text='EN', bg='orange', font=("Helvetica", 16))
    status_EN.grid(row=1, column=3)
    status_EN.grid_remove()
    status_VU = Message(conservation, text='VU', bg='yellow', font=("Helvetica", 16))
    status_VU.grid(row=1, column=4)
    status_VU.grid_remove()
    status_NT = Message(conservation, text='NT', bg='green', font=("Helvetica", 16))
    status_NT.grid(row=1, column=5)
    status_NT.grid_remove()
    status_LC = Message(conservation, text='LC', bg='blue', font=("Helvetica", 16))
    status_LC.grid(row=1, column=6)
    status_LC.grid_remove()
    conservation.grid_remove()

    mainloop()

def run_cv_demo():
    global animal
    currentZoom = 15
    curX = 0
    curY = 0
    curZ = 0
    curRotX = 0
    curRotY = 0
    curRotZ = 0
    anidex = 7
    #animal = loadAnimal(anidex)
    video = cv2.VideoCapture(1)
    ret, mtx, dist, rvecs, tvecs = calibrate(video)
    grabbed, frame = video.read()
    if not grabbed:
        print("failed")
        return

    height, width, _ = frame.shape

    frame_num = 1

    test = cv2.imread('checkerboard.jpg',0)
    ret, testcorners = cv2.findChessboardCorners(test, (8,6),None)

    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    while True:

        flags, hcursor, (x,y) = win32gui.GetCursorInfo()
        cv2.imshow('Live Session', frame)

        grabbed, frame = video.read()
        a, b, x, y, back, start, lt, rt, lb, rb, lt_x, lt_y, rt_x, rt_y  = getXbox();
        key = cv2.waitKey(1) & 0xFF
        # terminate session if esc or q was pressed
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (8,6),None)
        axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])


        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            frame = cv2.drawChessboardCorners(frame, (8,6), corners2, ret)
            ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)


            # project 3D points to image plane

            #axis2 = transformPointset(axis, TAAtoTM(np.array([x,y,-4,0,np.pi/2,np.pi/2])))
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            #imgpts2, jac2 = cv2.projectPoints(axis2, rvecs, tvecs, mtx, dist)

            #frame = draw(frame,corners2,imgpts2)
        #try:
            camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
            homography, mask = cv2.findHomography(testcorners, corners, cv2.RANSAC, 5.0)
            projection = projection_matrix(camera_parameters, homography)
            #try:
            if a == 1:
                curRotX = 0
                curRotY = 0
                curRotZ = 0
            currentZoom += rt_y/5
            if lt >= .2 or lt <= -.2:
                curRotX += lt_x
                curRotY += lt_y
                curRotZ += rt_x
            else:
                curX += lt_x*5
                curY += lt_y*5
                curZ += rt_x*5

            frame = render(frame, animal, projection, test, currentZoom, TAAtoTM(np.array([curX, curY, curZ, curRotX, curRotY, curRotZ])), False)
            #except:
            #    pass


        #except:
        #    pass

        if not grabbed or key == 27 or key == ord('q'):
            break

def setAniString(string):
    try:
        return OBJ(string + ".obj", swapyz=False)
    except:
        print("Failed to Find Match")


def loadAnimal(anidex):
    animal = OBJ("bear.obj", swapyz=True)
    if anidex == 1:
        animal = OBJ("bear.obj", swapyz=True)
    elif anidex == 2:
        nimal = OBJ("cat.obj", swapyz=True)
    elif anidex == 3:
        animal = OBJ("eagle.obj", swapyz=True)
    elif anidex == 4:
        animal = OBJ("elephant.obj", swapyz=True)
    elif anidex == 5:
        animal = OBJ("lizard.obj", swapyz=True)
    elif anidex == 6:
        animal = OBJ("shark.obj", swapyz=True)
    elif anidex == 7:
        animal = OBJ("whale.obj", swapyz=True)
    else:
        animal = OBJ("wolf.obj", swapyz=True)
    return animal

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img



def calibrate(video):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    Chess = False

    while not Chess:
        grabbed, img = video.read()
        if not grabbed:
            return
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (8,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            Chess = True
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (8,6), corners2,ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    print("ret")
    print(ret)
    print("mtx")
    print(mtx)
    print("dst")
    print(dist)
    return ret, mtx, dist, rvecs, tvecs

def transformPointset(points, tm):
    newpoints = np.zeros((np.shape(points)[0],3))
    for i in range(np.shape(points)[0]):
        tempM = np.array([[1, 0, 0, points[i,0]],[0, 1, 0, points[i,1]],[0, 0, 1, points[i,2]],[0, 0, 0, 1]])
        tempM2 = tm @ tempM
        newpoints[i,0] = tempM2[0,3]
        newpoints[i,1] = tempM2[1,3]
        newpoints[i,2] = tempM2[2,3]
    return newpoints

def stretchZSet(points, factor):
    newpoints = np.zeros((np.shape(points)[0],3))
    for i in range(np.shape(points)[0]):
        newpoints[i,0] = points[i,3]
        newpoints[i,1] = points[i,1]
        newpoints[i,2] = points[i,2]*factor
    return newpoints

def getXbox():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done=True

    pressed = controller.get_buttons() #You need this to register any buttons
    a = pressed[xbox360_controller.A] #Change Gaits
    b = pressed[xbox360_controller.B] #Toggle Walk Method
    x = pressed[xbox360_controller.X] #Does Nothing
    y = pressed[xbox360_controller.Y] #Balance Mode
    back = pressed[xbox360_controller.BACK]
    start = pressed[xbox360_controller.START] #Toggle Preset Heights
    lt = pressed[xbox360_controller.LEFT_BUMP] #Change Modes
    rt = pressed[xbox360_controller.RIGHT_BUMP] #Change Walk Parameters
    lb = pressed[xbox360_controller.LEFT_STICK_BTN]
    rb = pressed[xbox360_controller.RIGHT_STICK_BTN]
    lt_x, lt_y = controller.get_left_stick()
    rt_x, rt_y = controller.get_right_stick()
    du, dr, dd, dl = controller.get_pad()
    if lt_x < .1 and lt_x >-.1:
        lt_x = 0
    if lt_y < .1 and lt_y >-.1:
        lt_y = 0
    if rt_x < .1 and rt_x >-.1:
        rt_x = 0
    if rt_y < .1 and rt_y >-.1:
        rt_y = 0
    return a, b, x, y, back, start, lt, rt, lb, rb, lt_x, lt_y, rt_x, rt_y

def TAAtoTM(transaa):
    transaa = np.vstack((transaa[0],transaa[1],transaa[2],transaa[3],transaa[4],transaa[5]))
    s = transaa.shape[1]
    if s == 6:
        transaa = (transaa).conj().transpose()
    #
    mres = (mr.MatrixExp3(mr.VecToso3(transaa[3:6])))
    tm = np.vstack((np.hstack((mres,transaa[0:3])),np.array([0,0,0,1])))
    return tm

def render(img, obj, projection, model, currentZoom, rotmat, color=False):
    vertices = transformPointset(np.asarray(obj.vertices), rotmat)
    vertices = vertices.tolist()
    scale_matrix = np.eye(3) * currentZoom
    h, w = model.shape
    bcr = (190, 120, 40)
    i = bcr[0]
    j = bcr[1]
    k = bcr[2]
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:

            i+=1
            if i == bcr[0] + 55:
                i = bcr[0] - 55
                j+=1
                if j == bcr[1] + 15:
                    j = bcr[1] - 15
                    k+=1
                    if k == bcr[2] + 5:
                        k = bcr[2] - 5

            cv2.fillConvexPoly(img, imgpts, (i, k, j))


        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1] # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)


if __name__== '__main__':
    #startHere()
    Thread(target = run_cv_demo).start()
    Thread(target = UI).start()
