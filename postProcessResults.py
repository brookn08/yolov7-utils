import json

with open("./runs/test/yolov7_640_val9/yolov7_predictions.json") as f:
    data = f.read()

jsd = json.loads(data)

for pred in jsd:
    img_id = pred["image_id"]
    f = open(f"./detections/{img_id}.txt", "a")
    if pred["category_id"] == 0:
        cname = "kid"
    else:
        cname = "adult"

    conf = pred["score"]
    xt, yt, w, h = pred["bbox"]
    f.write(f"{cname} {conf} {xt} {yt} {w} {h}\n")
    