import numpy as np
import argparse
import imutils
import time
import cv2
import os


#Args
ap = argparse.ArgumentParser()
# ap.add_argument("--from_cam", default='yes', help="type 'no' and choose video to process ot type 'yes'(default) to use web cam stream")

ap.add_argument("-v", "--video", default=None, help="default(None) to use web cam stream ot type path to input video to"
                                                    " process selected file")
ap.add_argument("-o", "--output", default="output/output.avi", help="name and path to output video")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = ap.parse_args()

from_cam = True if args.video is None else False
labelsPath = "yolo-coco/coco.names"
weightsPath = "yolo-coco/yolov3.weights"
configPath = "yolo-coco/yolov3.cfg"

# parameters for subtitles
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 430)
fontScale = 0.8
fontColor = (255, 255, 255)
lineType = 2

LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

if from_cam:
    vs = cv2.VideoCapture(0)
else:
    vs = cv2.VideoCapture(args.video)
    prop = cv2.CAP_PROP_FRAME_COUNT

    total = int(vs.get(prop))
    print("There are " + str(total) + " frames in video")
writer = None
(W, H) = (None, None)


stats = []
while True:
    (grabbed, frame) = vs.read()

    if not grabbed:
        break
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    run_one_time = time.time() - start

    boxes = []
    confidences = []
    classIDs = []


    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > args.confidence:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # applying non-maxima suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args.confidence, args.threshold)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y, w, h) = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if not from_cam:
        if writer is None and not from_cam:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args.output, fourcc, 8, (frame.shape[1], frame.shape[0]), True)
            print("One frame took {:.4f} sec".format(run_one_time))
            print("Estimated total time to process video is {:.4f}".format(run_one_time * total))

        writer.write(frame)
    else:
        if len(classIDs) == 0:
            title = ("Here we can see nothing.")#.format(LABELS[label[0]]))
        else:
            string = ""
            for word in classIDs:
                string += LABELS[word]
                string += ", a "
            title = "Here we can see a {}".format(string[:-4] + ".")

        cv2.putText(frame, title, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        cv2.imshow('Stream IP Camera OpenCV', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    stats.append(classIDs)

if writer is not None:
    writer.release()
vs.release()
cv2.destroyAllWindows()
