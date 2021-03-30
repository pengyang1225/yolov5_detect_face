"""
convert annotations to yolo format
"""
import os
from pathlib import Path
from PIL import Image
import csv
import sys
import glob



def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2] / 2) * dw
    y = (box[1] + box[3] / 2) * dh
    w = box[2] * dw
    h = box[3] * dh
    return (x, y, w, h)
            

train_file = 'images.txt'  
train_file_txt = ''

root = sys.argv[1]
ann_d = os.path.join(root, 'annotations')
img_d = os.path.join(root, 'images')
labels_d = os.path.join(root, 'labels')
os.makedirs(labels_d, exist_ok=True)
anns = glob.glob(os.path.join(ann_d, '*.txt'))
for ann in anns:
    ans = ''
    outpath = os.path.join(labels_d, os.path.basename(ann))
    img_file = os.path.join(img_d, os.path.splitext(os.path.basename(ann))[0] + '.jpg')
    with Image.open(img_file) as Img:
        img_size = Img.size
    with open(ann) as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if row[4] == '0':
                continue
            bb = convert(img_size, tuple(map(int, row[:4])))
            ans = ans + str(int(row[5])-1) + ' ' + ' '.join(str(a) for a in bb) + '\n'
            with open(outpath, 'w') as outfile:
                outfile.write(ans)
    # train_file_txt = train_file_txt + wd + '/images/' + ann[:-3] + 'jpg\n'

print('done')
# with open(train_file, 'w') as outfile:
#     outfile.write(train_file_txt)
