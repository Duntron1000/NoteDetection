import cv2 as cv

vid = cv.VideoCapture(0)

while(True):
    ret, src = vid.read()

    cv.imshow('frame', src)


    blur = cv.GaussianBlur(src,  (13,13), 0)
    cv.imshow("blur", blur)

    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (0, 150, 140), (12, 255, 255))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, (11, 11), iterations=10)
    cv.imshow("mask", mask)
    masked = cv.bitwise_and(src, src, mask=mask)
    cv.imshow("masked", masked)

    canny = cv.Canny(mask, 75, 100)
    cv.imshow("canny", canny)

    contours, hiarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_list = []
    area_list = []
    aprox_list = []
    centers_list = []

    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, False), False)
        area = cv.contourArea(contour)
        if (len(approx) > 10) and (area > 100):
            contours_list.append(contours)
            aprox_list.append(approx)
            area_list.append(area)
    

    print("Contours List" + str(contours_list))
    print("Area List" + str(area_list))
    print("Centers List" + str(centers_list))

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()
