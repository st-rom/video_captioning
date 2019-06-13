import numpy as np
import cv2


def main():
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter('input/input.avi', fourcc, 8, (640,480), True)
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Stream IP Camera OpenCV', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
