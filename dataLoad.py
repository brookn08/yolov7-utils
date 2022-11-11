import os
from dataset_loader import DatasetLoader

def xyxy2xywh(x):
    y = [0, 0, 0, 0]
    y[0] = (x[0] + x[2]) / 2  # x center
    y[1] = (x[1] + x[3]) / 2  # y center
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height
    return y

dl = DatasetLoader("/home/akshay_goindani/data_akshay_goindani/kid_adult_detection/data/")
print(len(dl))

f = open("./train.txt", "w")

for _, batch in enumerate(dl):
    img_path, bboxes, shape = batch
    name, ext = str(img_path).split(".")
    dirn, filn = str(img_path).split("/images/")
    os.system(f"mkdir -p {dirn}/labels/")
    lab_file = str(name).replace("/images/", "/labels/") + ".txt"

    h, w, c = shape

    with open(lab_file, "w") as f1:
        for box in bboxes:
            # normalizing 
            box[0] /= w
            box[1] /= h
            box[2] /= w
            box[3] /= h
            nbox = xyxy2xywh(box)

            if box[-1] == "kid_body":
                f1.write(f"0 {nbox[0]} {nbox[1]} {nbox[2]} {nbox[3]}\n")
            else:
                f1.write(f"1 {nbox[0]} {nbox[1]} {nbox[2]} {nbox[3]}\n")

    f.write(str(img_path).strip() + "\n")

f.close()
