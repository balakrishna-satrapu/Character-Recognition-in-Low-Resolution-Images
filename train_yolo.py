%cd /content/drive/MyDrive/yolov5
!python train.py \
  --imgsz 144 \
  --batch-size 32 \
  --epochs 100 \
  --data /content/drive/MyDrive/yolov5/yolo_char_rec/datasets/plate.yaml \
  --cfg /content/drive/MyDrive/yolov5/yolo_char_rec/models/yolov5_character.yaml \
  --weights '' \
  --name yolo_char_rec \
  --rect