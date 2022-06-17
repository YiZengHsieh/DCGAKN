import sys
import argparse
from yolo_height import YOLO, detect_video
from PIL import Image
import glob
from keras import backend as K
import os
import cv2
import time
K.clear_session()
'''
#若要改回一張一張圖片圖取的樣子，請把這邊註解拿掉，並將下方丟入影片的部分註解
def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()
'''
def video_cut(input_path):
    vc = cv2.VideoCapture(input_path) #讀入影片
    c=1
    if vc.isOpened(): #check
        rval , frame = vc.read()
        print("video frame is check")
    else:
        rval = False
        print("video frame is not ok")
     
    timeF = 1  #要壓縮的影片FPS
     
    while rval:   #循環讀取
        rval, frame = vc.read()
        if(c%timeF == 0):
            cv2.imwrite('J_input/'+str(c) + '.jpg',frame) #存為圖像
        c = c + 1
        cv2.waitKey(1)
    vc.release()

def video_mix(outdir,output_path):
    #img path
    im_dir = outdir
    #output path
    video_dir = output_path
    #fps
    fps = 4
    #img value
    num = len(os.listdir(im_dir))
    #img size
    img_size = (512,424)

    #fourcc = cv2.cv.CV_FOURCC('M','J','P','G')#這是opencv2.4用法
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #這是opencv3.0用法
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

    for i in range(1,num):
        im_name = os.path.join(im_dir, str(i).zfill(1)+'.jpg')
        frame = cv2.imread(im_name)
        videoWriter.write(frame)
        print (im_name)
    videoWriter.release()
    print ('mix finish....')

def detect_img(yolo):
    #input_video='input_video/move2.mp4' #預想丟入測試的影片
    #output_path='C:/Users/hpz640/Desktop/YOLOv3/keras-yolo3-master/output_video/mmm8.avi' #輸出路徑+影片名稱，我當初做都是以avi為主
    #video_cut(input_video) #將影片分割
    path = "J_input/*.jpg"  #設定輸入圖片路徑
    outdir = "J_output" #設定輸出圖片路徑
    i=1
    for jpgfile in glob.glob(path):
        img = Image.open(jpgfile)
        img = yolo.detect_image(img,i)
        img.save(os.path.join(outdir, os.path.basename(jpgfile)))
        i+=1
    #video_mix(outdir,output_path) #將預測多張圖片壓縮成影片
    yolo.close_session() #結束

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
