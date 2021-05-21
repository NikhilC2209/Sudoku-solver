import cv2
import numpy as np
import matplotlib.pyplot as plt


##### 1. Preprocessing image
def pre_process(img):
    cv2.imshow('Image2',img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #### In case of colored images to gray
    img = cv2.GaussianBlur(img,(5,5),1)
    imgThreshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    process = cv2.bitwise_not(imgThreshold, imgThreshold)   ### INVERT COLORS TO BINARY IMAGE
    cv2.imshow('Image4',imgThreshold)
    return process

def find_corners(img):
    ext_contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    ### RETR_EXTERNAL Returns only outermost contours all child contours are ignored
    #cv2.drawContours(img, ext_contours, -1, (0, 255, 0), 3)
    #cv2.imshow('cont',img)
    #print(ext_contours)
    ext_contours = ext_contours[0] if len(ext_contours) == 2 else ext_contours[1]
    ext_contours = sorted(ext_contours, key=cv2.contourArea, reverse=True)    
    for c in ext_contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        #cv2.imshow('sudoku',approx)
        if len(approx) == 4:
            return approx

def order_points(corners):
    i=0
    corners2=[(),(),(),()]        ##### Initialize with empty tuples
    for corner in corners:
        corners2[i]=(corner[0][0],corner[0][1])
        i+=1      
    top_r, top_l, bottom_l, bottom_r = corners2[0], corners2[1], corners2[2], corners2[3]
    return top_l, top_r, bottom_r, bottom_l            

def transformimg(image,corners):
    ordered_corners = order_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    width_bottom = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))    ### FIND TOP & BOTTOM WIDTH OF SUDOKU GRID
    width_top = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_bottom), int(width_top))

    height_top = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))            ### FIND TOP & BOTTOM WIDTH OF SUDOKU GRID 
    height_bottom = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_top), int(height_bottom))

    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],[0, height - 1]], dtype="float32")
    ordered_corners = np.array(ordered_corners, dtype="float32")
    grid = cv2.getPerspectiveTransform(ordered_corners, dimensions)
    #print(type(grid))
    return cv2.warpPerspective(image, grid, (width, height))

def create_image_grid(img):
    grid = np.copy(img)
    # not all sudoku out there have same width and height in the small squares so we need to consider differnt heights and width
    edge_h = np.shape(grid)[0]
    edge_w = np.shape(grid)[1]
    celledge_h = edge_h // 9
    celledge_w = np.shape(grid)[1] // 9

    grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding the cropped grid and inverting it
    grid = cv2.bitwise_not(grid, grid)


    tempgrid = []
    for i in range(celledge_h, edge_h + 1, celledge_h):
        for j in range(celledge_w, edge_w + 1, celledge_w):
            rows = grid[i - celledge_h:i]
            tempgrid.append([rows[k][j - celledge_w:j] for k in range(len(rows))])

    # Creating the 9X9 grid of images
    finalgrid = []
    for i in range(0, len(tempgrid) - 8, 9):
        finalgrid.append(tempgrid[i:i + 9])

    # Converting all the cell images to np.array
    for i in range(9):
        for j in range(9):
            finalgrid[i][j] = np.array(finalgrid[i][j])

    try:
        for i in range(9):
            for j in range(9):
                np.os.remove("BoardCells/cell" + str(i) + str(j) + ".jpg")      ### REMOVE EXISTING IMAGES if any
    except:
        pass
    try:    
        for i in range(9):
            for j in range(9):
                cv2.imwrite(str("BoardCells/cell" + str(i) + str(j) + ".jpg"), finalgrid[i][j])     ### WRITE THE NEW CELLS OF SUDOKU as images
    except:
        print("err here")            

    return finalgrid

def scale_and_centre(img, size, margin=20, background=0):
    h, w = img.shape[:2]

    def centre_pad(length):
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))



def extract():
    img = cv2.imread('assets/sudoku_1.jpg',1)
    processed_img = pre_process(img)
    sudoku = find_corners(processed_img)
    transformed_img = transformimg(img,sudoku)
    cv2.imshow('sudoku',transformed_img)
    cropped = 'assets/cropped_img.png'
    final_sudoku_grid = 'assets/final_sudoku.png'
    cv2.imwrite(cropped, transformed_img)
    final_grid = create_image_grid(transformed_img)
    return final_grid

final_grid=extract()
cv2.waitKey()
cv2.destroyAllWindows()