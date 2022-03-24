import cv2
import time
import os
import tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)

class detector:
    def __init__(self):
        pass

    def read_classes(self, class_filepath):
        with open(class_filepath,'r') as f:
            self.class_list = f.read().splitlines()

        # unique colors for each labels
        self.color_list = np.random.uniform(low = 0, high=255, size=(len(self.class_list), 3))

        # print(len(self.class_list), len(self.color_list))

    def downloadModel(self, modelURL):
        filename = os.path.basename(modelURL)
        
        self.modelname = filename[:filename.index('.')]

        self.cacheDir = './pretrained_models'

        os.makedirs(self.cacheDir, exist_ok=bool)

        get_file(fname=filename, origin = modelURL, cache_dir = self.cacheDir,
        cache_subdir = "checkpoints", extract = True)

    def loadModel(self):
        print("Loading model "+self.modelname)

        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelname, "saved_model"))

        print("Model is loaded successfuly....")
    
    def createBoundingBox(self, image, threshold = 0.5):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis]

        detections = self.model(inputTensor)
        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape

        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size = 50, 
        iou_threshold = threshold, score_threshold = threshold)

        print(bboxIdx)

        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100*classScores[i])
                classIndex = classIndexes[i]

                classLabelText = self.class_list[classIndex]
                classColor = self.color_list[classIndex]

                displayText = '{}: {}%'.format(classLabelText, classConfidence)

                ymin, xmin, ymax, xmax = bbox

                xmin, xmax, ymin, ymax = (xmin*imW, xmax*imW, ymin*imH, ymax*imH)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color = classColor, thickness = 1)

                cv2.putText(image, displayText, (xmin, ymin-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
                

        return image



    def predictImage(self, imagePath, threshold):
        image = cv2.imread(imagePath)
        
        bboxImage = self.createBoundingBox(image, threshold)
        
        cv2.imwrite(self.modelname + ".jpg", bboxImage)
        cv2.imshow("Result",bboxImage)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predictVideo(self, videoPath, threshold = 0.5):
        cap = cv2.VideoCapture(videoPath)
        if cap.isOpened() == False:
            print("Error opening the video file")
            return

        success, image = cap.read()
        startTime = 0

        while success:
            currentTime = time.time()

            fps = 1/(currentTime - startTime)
            startTime = currentTime

            bboxImage = self.createBoundingBox(image, threshold)

            cv2.putText(bboxImage, "FPS: "+str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
            cv2.imshow("Result_video", bboxImage)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            success,image = cap.read()

        cv2.destroyAllWindows()