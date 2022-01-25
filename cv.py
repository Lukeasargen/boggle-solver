import cv2
import numpy as np
from PIL import Image

image_path = "images/IMG_5007.JPG"
# rgb = cv2.imread(image_path)
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
w,h,c = image.shape
size = 700
if w>h:
    width = size; height = int(h/w*size)
else:
    height = size; width = int(w/h*size)
image = cv2.resize(image, (height, width), interpolation=cv2.INTER_LINEAR)
display = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Remove noise with a gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# Adaptive thresholding
threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 7)
# # Find contours
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Find the blob with the biggest area
AREA_THRESHOLD = 1200
biggest = None
max_area = 0
for i in contours:
    area = cv2.contourArea(i)
    if area > AREA_THRESHOLD:
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02*peri, True)
        if area > max_area and len(approx) == 4:
            biggest = approx
            max_area = area
            best_cont = i
        cv2.polylines(display, [approx], True, (0, 0, 255), 3)
if max_area > 0:
    cv2.polylines(display, [biggest], True, (0, 255, 0), 3)

tl, tr, br, bl = biggest[0][0], biggest[1][0], biggest[2][0], biggest[3][0]
pts1 = np.float32([tl,tr,bl,br])
output_size = 400
pts2 = np.float32([[0,0], [output_size,0],[0,output_size],[output_size,output_size]])


M = cv2.getPerspectiveTransform(pts1,pts2)
warp_color = cv2.warpPerspective(image,M,(output_size,output_size))
warp_gray = cv2.warpPerspective(gray,M,(output_size,output_size))
warp_blur = cv2.warpPerspective(blur,M,(output_size,output_size))
warp_threshold = cv2.warpPerspective(threshold,M,(output_size,output_size))


square_dim = 6
diff = int(output_size/square_dim)
buffer = 10  # extra pixels around the letter

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
custom_config = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz --psm 6 min_characters_to_try=10'

for x in range(square_dim):
    for y in range(square_dim):
        i = diff*x
        j = diff*y
        roi = warp_threshold[max(j-buffer,0):min(j+diff+2*buffer,output_size), max(i-buffer,0):min(i+diff+2*buffer,output_size)]
        # cv2.imshow('roi', roi); cv2.waitKey(0)
        roi_pil = Image.fromarray(roi)
        # s = pytesseract.image_to_string(roi_pil, config=custom_config)
        # print(f"OCR ({x},{y}):", s)
        # contours, hierarchy = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours))
        # for cnt in contours:
        #     if cv2.contourArea(cnt) > 20:
        #         [x, y, w, h] = cv2.boundingRect(cnt)
        #         print(x,y,w,h)
        #         cv2.rectangle(warp_gray, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # break
    break


cv2.imshow("Image", display)
cv2.imshow("Threshold", threshold)
cv2.imshow('Warp', warp_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()

