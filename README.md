总之是一个独立出来的人脸识别模块，把两张需要识别的人脸图像放在./image/目录下，命名为photo.png和request_photo.jpeg
需要安装python 3.6.7 版本，运行pip install -r .\requirements.txt安装依赖，然后python .\Recognize.py运行项目
有可能dlib不好安装,总之我还是有编译好的dlib版本= =,可以直接用
更新了动作识别模块，其中通过cv2.VideoCapture模块读取image目录下的movie，这里可以根据需要更改目录
模块文件命名是GetCsv，至于为啥用这样一个毫无关系的名字，只能说是有历史遗留问题
