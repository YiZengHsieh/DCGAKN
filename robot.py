import subprocess
subprocess.call(["python3", "E:/python/KinectV2/KinectV2-master/takePictures.py"])
subprocess.call(["python3", "yolo_video_step.py","--image"])
subprocess.call(["python3", "reader.py"])
subprocess.call(["python3", "load_generator.py"])