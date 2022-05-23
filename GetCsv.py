#%%
import os
import math
import numpy as np
import json
import torch
# %%
import cv2

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("./video.MP4", fourcc, 30.0, (1080, 1920))
# %%
import face_recognition
from PIL import Image, ImageDraw

global numofall
global numoftrue
global numofimage
global eyelen
global leftrighteyelen
global updownlen

numofall = 0
numofimage = 0
numoftrue = 0
eyelen = 0
leftrighteyelen = 0
updownlen = 0
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
#%%
def CalDistance(pointleft, pointright):
    distance = math.pow(math.pow(pointleft[0] - pointright[0], 2) + math.pow(pointleft[1] - pointright[1], 2), 0.5)
    return distance
# %%
def CalAround(pointleft, pointright, length, rate):
    distance = CalDistance(pointleft, pointright) / rate
    return math.acos(distance / length) * 180 / 3.14
# %%
def CalLeftRight(pointleft, pointright):
    return math.atan2(pointright[1] - pointleft[1], pointright[0] - pointleft[0]) * 180 / 3.14

# %%
def CalUpDown(pointup, pointdown, length, rate):
    distance = CalDistance(pointup, pointdown) / rate
    return math.acos(distance / length) * 180 / 3.14
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
        pil_image = Image.fromarray(imagefirst)
        d = ImageDraw.Draw(pil_image)

        # print(imagelist)

        # for face_landmarks in imagelist:

        #         for facial_feature in face_landmarks.keys():
        #             d.line(face_landmarks[facial_feature], width=5)

        line = []
        line.append(CalCenter(imagelist[0]['left_eye']))
        line.append(CalCenter(imagelist[0]['right_eye']))

        # d.line(line, width=5)

        line2 = []
        # line2.append(imagelist[0]['nose_bridge'][0])
        line2.append(imagelist[0]['nose_bridge'][len(imagelist[0]['nose_bridge']) - 1])
        
        line3 = []
        line3 = imagelist[0]['top_lip'] + imagelist[0]['bottom_lip']
        lengthlips = (CalCenter(line3))

        line2.append(lengthlips)

        # d.line(line2, width=5)

        lengtheyes = (CalAroundLength(imagelist[0]['left_eye']) + CalAroundLength(imagelist[0]['right_eye'])) / 2
        rate = 1

        global eyelen
        if eyelen == 0:
            eyelen = lengtheyes
        else:
            rate = lengtheyes / eyelen

        global leftrighteyelen
        if leftrighteyelen == 0:
            leftrighteyelen = CalDistance(line[0], line[1])

        global updownlen
        if updownlen == 0:
            updownlen = CalDistance(line2[0], line2[1])

        # eye_length = CalDistance(line[0], line[1]) / rate
        # nose_length = CalDistance(line2[0], line2[1]) / rate
        around = CalLeftRight(line[0], line[1])

        leftrightaround = CalAround(line[0], line[1], leftrighteyelen , rate)
        updownaround = CalUpDown(line2[0], line2[1], updownlen, rate)

        print('leftrightaround:{}'.format(leftrightaround))
        print('updownaround:{}'.format(updownaround))
        print('headaround:{}'.format(CalLeftRight(line[0], line[1])))

        # img = cv2.cvtColor(np.asarray(pil_image),cv2.COLOR_RGB2BGR)  
        # out.write(img)

        # pil_image.show()

        res = [leftrightaround, updownaround, around]

        return res
# %%
def FaceRecognize(know_im, imagefirst):
    # know_im = face_recognition.load_image_file("./image/image_true/doggy_true.jpg")
    know_encodings = face_recognition.face_encodings(know_im)
    first_encodings = face_recognition.face_encodings(imagefirst)
    global numofall
    numofall = numofall + 1
    if len(know_encodings) == 0:
        global numoftrue
        numoftrue = numoftrue + 1
        return 'Error, no person in know_encoding'
    elif len(first_encodings) == 0:
        global numofimage
        numofimage = numofimage + 1
        return 'Error, no person in this image'
    else:
        return face_recognition.compare_faces([know_encodings[0]], first_encodings[0])[0]

# %%
# def CreateNewJson(num):
#     filename='data.json'
#     with open('data.json','w') as da:
#         json.dump(num,da)
#     return num

# %%
filename = './image'
flag = 0
# net = torch.load('./model_action/action.tar')

for dirname in os.listdir(filename):
    # print(dirname)
    # print(os.path.join(filename, dirname + '/movie'))
    # print(os.listdir(os.path.join(filename, dirname + '/movie')))
    thisdir = os.path.join(filename, dirname)
    print(thisdir)
    movdir = os.listdir(os.path.join(filename, dirname + '/movie'))
    # print(movdir[0])
    # if 1 == flag:
    try:
        know_im = face_recognition.load_image_file(os.path.join(thisdir, 'photo.png'))
        imagefirst = face_recognition.load_image_file(os.path.join(thisdir, 'request_photo.jpeg'))
    except Exception as e:
        print(e)
    else:
        gray = cv2.cvtColor(imagefirst, cv2.COLOR_BGR2GRAY)
        gamma = CalGamma(JudgeBrightness(gray))
        img_gamma = GammaTransformation(imagefirst, gamma)
        print(str(FaceRecognize(know_im, np.uint8(img_gamma))))
    # flag = 1

# %%

# print('all:{}, noknow:{}, noimage:{}, rate:{}'.format(numofall, numoftrue, numofiamge, (numofiamge + numoftrue) / numofall))
    for movname in movdir:
        # print(movname)
        video_capture = cv2.VideoCapture(os.path.join(thisdir, 'movie/' + movname))
        count = 0
        countres = 0
        # facedata = []
        # flagres = 0
        # jso = [{'x': [], 'y': []}]
        while True:
            # print(1)
            ret,frame = video_capture.read()
            # unknown_image = face_recognition.load_image_file(frame)
            try:
                frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # print(2)
            except:
                print(1)
                break
                
            # frame = frame[:, :, ::-1]

            # frame = cv2.flip(frame, 0)
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # face_landmarks_list = face_recognition.face_landmarks(frame)
            # pil_image = Image.fromarray(frame)
            # pil_image.show()
            # d = ImageDraw.Draw(pil_image)

            # print(face_landmarks_list)
            if count % 30 == 0:
                try:
                    pil_image = Image.fromarray(frame)
                    img = cv2.cvtColor(np.asarray(pil_image),cv2.COLOR_RGB2BGR)
                        # pil_image = Image.fromarray(img)
                        # pil_image.show()

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gamma = CalGamma(JudgeBrightness(gray))
                    frame = GammaTransformation(img, gamma)

                    res = ImageShow(np.uint8(frame))
                    print(FaceRecognize(know_im, np.uint8(frame)))

                    print(net(res))
                except Exception as e:
                    print(e)
                    continue
            count = count + 1

        # out.release()
        video_capture.release()
        cv2.destroyAllWindows()

# %%