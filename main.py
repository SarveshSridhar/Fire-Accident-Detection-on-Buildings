from detector import *


modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz'
# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"

classFile = "coco.names"
imagePath = "test/4.jpg"
threshold = 0.5
videoPath = "test/test_video1.mp4"


detector = detector()
detector.read_classes(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
# detector.predictImage(imagePath, threshold)
detector.predictVideo(videoPath, threshold)