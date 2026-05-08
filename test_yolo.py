%cd /content/drive/MyDrive/yolov5
!python detect.py \
    --weights runs/train/yolo_char_rec10/weights/best.pt \
    --imgsz 144 \
    --conf-thres 0.5 \
    --iou-thres 0.3 \
    --source /content/00VZ7H.jpg \
    --name yolo_char_rec_detect