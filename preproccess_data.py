import cv2
import os
import csv

img = cv2.imread("visikom_1/training/0/1.png", cv2.IMREAD_GRAYSCALE)
extra_columns=[0,0,0,0,0,0,0,0,0,0]
with open("images_with_extra.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for i in range(10):
        print(i)
        folder_path = f"visikom_1/training/{i}"
        for filename in os.listdir(folder_path):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                flat_img= img.flatten()/255.0
                extra_columns[i]=1
                row_data = flat_img.tolist()+ extra_columns 
                extra_columns[i]=0
                writer.writerow(row_data)
