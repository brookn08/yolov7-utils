import cv2

with open("../kid_adult_detection/test.txt") as f:
    files = f.readlines()

for fl in files:
    img = cv2.imread(fl.strip())
    h, w, c = img.shape

    label_fl = fl.strip().replace("/images/", "/labels/")
    label_fl = label_fl.replace(".jpg", ".txt")
    label_id = label_fl.split("/labels/")[-1]

    f1 = open(f"./groundtruths/{label_id}", "w")

    with open(label_fl) as f:
        for line in f:
            clss, xc, yc, bw, bh = line.strip().split()
            
            xc = float(xc)
            yc = float(yc)
            bw = float(bw)
            bh = float(bh)

            if clss == "0":
                clss = "kid"
            else:
                clss = "adult"

        xc, yc, bw, bh = xc * w, yc * h, bw * w, bh * h
        xt = xc - bw/2
        yt = yc - bh/2
        
        f1.write(f"{clss} {xt} {yt} {bw} {bh}\n")
    