import cv2
from ultralytics import YOLO
import argparse
import numpy as np
import supervision as sv


ZONE_POLYGONE = np.array([
    [0,0],
    [1280,0],
    [1280,720],
    [0,720]
])
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='yolov8')
    parser.add_argument('--webcam-resolution',
                        default = [1280,720],
                        nargs = 2,
                        type = int,
                        help = 'Resolution of the webcam feed(width heigth)')
    args = parser.parse_args()
    return args
def main():
    args = parse_arguments()
    frame_width,frame_height = args.webcam_resolution
    
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
    
    model = YOLO('yolov8s.pt')
    
    box_annotate = sv.BoxAnnotator(
    thickness=1,        # Thickness of the bounding box
    color= sv.Color.RED
    )
    zone = sv.PolygonZone(polygon=ZONE_POLYGONE)
    zone_annotator = sv.PolygonZoneAnnotator(zone = zone, color = sv.Color.RED)
    while True:
        ret,frame = cam.read()
        
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.class_id ==0]
        detections.labels = [
            f'{model.model.names[class_id]} {confidence:.2f}'
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        frame = box_annotate.annotate(scene = frame, detections= detections)
        in_zone = zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)
        cv2.imshow('yolov8',frame) 
    
        
        if (cv2.waitKey(30) == 27): 
            break
        
    
if __name__ =='__main__':
    main()