# prepare

```
python3 convert_2_yolo.py ./VisDrone2019-DET-train
```

soft link root dir which contains `VisDrone2019-DET-train` and `VisDrone2019-VID-val` under `data`.

We mainly train VisDrone on yolov6 which is ghostnet with yolov5 with SPP support enabled.

We run:

```
python3 train_v6.py --noautoanchor --data ./mana_exp/visdrone.yaml --cfg ./models/yolov6_ghostnet_visdrone.yaml
```