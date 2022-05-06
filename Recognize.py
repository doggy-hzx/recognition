#%%
import os
import math
import numpy as np
# %%
import cv2

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("./video.MP4", fourcc, 30.0, (1080, 1920))
# %%
import face_recognition
from PIL import Image, ImageDraw

# %%
def JudgeBrightness(image):
	sum = 0
	count = 0
	for i in image:
		for j in i:
			sum = sum + j
			count = count + 1
	return sum / count

# %%
def CalGamma(average):
	return math.log(0.5, average / 255)

# %%
def GammaTransformation(image, gamma):
	image_cp = np.copy(image)
	output_imgae = 255 * np.power(image_cp.astype(int) / 255, gamma)
	return output_imgae

def CalDistance(pointleft, pointright):
    distance = math.pow(math.pow(pointleft[0] - pointright[0], 2) + math.pow(pointleft[1] - pointright[1], 2), 0.5)
    return distance
# %%
def CalAround(pointleft, pointright, length):
    distance = CalDistance(pointleft, pointright)
    return math.acos(distance / length) * 180 / 3.14
# %%
def CalLeftRight(pointleft, pointright):
    return math.atan2(pointright[1] - pointleft[1], pointright[0] - pointleft[0]) * 180 / 3.14

# %%
def CalUpDown(pointup, pointdown, length):
    distance = abs(pointdown[1] - pointup[1])
    return math.asin(distance / length) * 180 / 3.14
# %%
def CalCenter(pointgroup):
    x = y = n = 0
    for point in pointgroup:
        x += point[0]
        y += point[1]
        n += 1
    return (x / n, y / n)
# %%
def CalAroundLength(pointgroup):
    length = 0
    p = pointgroup[0]
    flag = 0
    for point in pointgroup:
        if flag == 1:
            plen = CalDistance(p, point)
            length = length + plen

        else:
            flag = 1
    return length

# %%
# imagefirst = face_recognition.load_image_file("./image/IMG_2100.jpg")
def ImageShow(imagefirst):
    imagelist = face_recognition.face_landmarks(imagefirst)
    # print(imagelist)
    if (len(imagelist) == 0):
        return
    else:
        imgroup = []
        for im in imagelist:
            # pil_image = Image.fromarray(im)
            lengtheyes = (CalAroundLength(im['left_eye']) + CalAroundLength(im['right_eye'])) / 2
            rate = 1

            # global eyelen
            # if eyelen == 0:
            #     eyelen = lengtheyes
            # else:
            #     rate = lengtheyes / eyelen


            line = []
            line.append(CalCenter(im['left_eye']))
            line.append(CalCenter(im['right_eye']))

            line2 = []
            line2.append(im['nose_bridge'][0])
            line2.append(im['nose_bridge'][len(im['nose_bridge']) - 1])

            imgg = []
            imgg.append(CalDistance(line[0], line[1]))
            imgg.append(CalDistance(line2[0], line2[1]))
            imgg.append(CalLeftRight(line[0], line[1]))
            # print('eye_length:{}'.format(CalDistance(line[0], line[1])))
            # print('nose_length:{}'.format(CalDistance(line2[0], line2[1])))
            # print('around:{}'.format(CalLeftRight(line[0], line[1])))
            imgroup.append(imgg)
        # img = cv2.cvtColor(np.asarray(pil_image),cv2.COLOR_RGB2BGR)  
        # out.write(img)

        # pil_image.show()

        return imgroup

# %%
def FaceRecognize(know_im, imagefirst):
    # know_im = face_recognition.load_image_file("./image/image_true/doggy_true.jpg")
    know_encodings = face_recognition.face_encodings(know_im)
    first_encodings = face_recognition.face_encodings(imagefirst)
    if len(know_encodings) == 0:
        return 'Error, no person in know_encoding'
    elif len(first_encodings) == 0:
        return 'Error, no person in this image'
    else:
        trueandfalse = []
        for first_encode in first_encodings:
            flag = 0
            for know_encode in know_encodings:
                if face_recognition.compare_faces([know_encode], first_encode)[0]:
                    flag = 1
            if flag == 0:
                trueandfalse.append("false")
            else:
                trueandfalse.append("true")
        # return face_recognition.compare_faces([know_encodings[0]], first_encodings[0])[0]
        return trueandfalse
# %%
try:
    know_im = face_recognition.load_image_file('./image/photo.png')
    imagefirst = face_recognition.load_image_file('./image/request_photo.jpeg')
except Exception as e:
    print(e)
else:
    gray = cv2.cvtColor(imagefirst, cv2.COLOR_BGR2GRAY)
    gamma = CalGamma(JudgeBrightness(gray))
    img_gamma = GammaTransformation(imagefirst, gamma)
    print(str(FaceRecognize(know_im, np.uint8(img_gamma))))
    print(ImageShow(np.uint8(img_gamma)))

# %%
# try:
#     frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
# except Exception as e:
# 	print(e)

# %%
# try:
#     pil_image = Image.fromarray(frame)
#     img = cv2.cvtColor(np.asarray(pil_image),cv2.COLOR_RGB2BGR)
#     # pil_image = Image.fromarray(img)
#     # p
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gamma = CalGamma(JudgeBrightness(gray))
#     frame = GammaTransformat
#     ImageShow(np.uint8(frame))
#     print(FaceRecognize(know_im, np.uint8(frame)))
# except Exception as e:
#     print(e)