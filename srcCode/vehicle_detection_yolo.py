# Script usage: python vehicle_detection_yolo.py --video==traffic.mp4 --device '<cpu/gpu>'
#               python vehicle_detection_yolo.py --image==cars.jpg --device '<cpu/gpu>'

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
# Parameters intialization 
conf_thres = 0.5  #Confidence threshold
iou_thres = 0.4   #Non-maximum suppression threshold
img_width = 416       #Width of network's input image
img_height = 416      #Height of network's input image

parse = argparse.ArgumentParser(description='Vehicle Detection using YOLO in OPENCV')
parse.add_argument('--device', default='cpu', help="perferable inference device 'cpu' or 'gpu'.")
parse.add_argument('--image', help='Path to image file.')
parse.add_argument('--video', help='Path to video file.')
args = parse.parse_args()
        
# Load names of classes
classFile = "coco.names"
classes = None
with open(classFile, 'rt') as cfile:
    classes = cfile.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfig = "yolov3.cfg"
modelWeights = "yolov3.weights"

yolonet = cv.dnn.readNetFromDarknet(modelConfig, modelWeights)

if(args.device == 'cpu'):
    yolonet.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    yolonet.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    print('Using CPU device.')
elif(args.device == 'gpu'):
    yolonet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    yolonet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print('Using GPU device.')

# Get the names of the output layers
def getOutputsNames(yolonet):
    # Get the names of all the layers in the network
    layersNames = yolonet.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[int(i) - 1] for i in yolonet.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Get the label for the class name and its confidence
    conflabel = '%.2f' % conf
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)


    if classes:
        assert(classId < len(classes))
        conflabel = '%s:%s' % (classes[classId], conflabel)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(conflabel, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, conflabel, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocessor(frame, outs):

    frameHeight,frameWidth = frame.shape[0],frame.shape[1]
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    confidences = []
    clsIds = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = np.array([detection[6],detection[7],detection[8],detection[10],detection[12]])
            clsId = np.argmax(scores)
            confidence = scores[clsId]
            if confidence > conf_thres:
                Bw = int(detection[2] * frameWidth)
                Bh = int(detection[3] * frameHeight)
                Bx = int(detection[0] * frameWidth)
                By = int(detection[1] * frameHeight)
                left = int(Bx - Bw / 2)
                top = int(By - Bh / 2)
                confidences.append(float(confidence))
                clsIds.append(clsId)
                boxes.append([left, top, Bw, Bh])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_thres, iou_thres)
    for i in indices:
        i = int(i)
        box = boxes[i]
        left,top,Bw,Bh = box
        drawPred(clsIds[i], confidences[i], left, top, left + Bw, top + Bh)
    return indices.shape[0]
# Process inputs

winName = 'Deep learning vehicle detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outFile = args.video[:-4]+'_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))


while cv.waitKey(1) < 0:
    
    # get frame from the video
    hasFrame, frame = cap.read()
    
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break
    
    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (img_width, img_height), [0,0,0], 1, crop=False)

    # Sets the input to the network
    yolonet.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = yolonet.forward(getOutputsNames(yolonet))

    # Remove the bounding boxes with low confidence
    id = postprocessor(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = yolonet.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    numcars = 'Vehicles in frame: %d' % id
    cv.putText(frame, numcars, (0,30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outFile, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))

    cv.imshow(winName, frame)

