import os
import tkinter as tk
from PIL import ImageTk

MTSD_FULLY_ANNOTATED_CLASSIFIED_CROPPED_IMAGES_TRAIN_DIR="../MTSD/mtsd_fully_annotated_classified_cropped_images_train/"

def imshow(image_dir, MTSD_classes, idx):
    cls_img_dir = image_dir + MTSD_classes[idx] + '/'
    image_name = os.listdir(path=cls_img_dir)
    image_file = cls_img_dir + image_name[0]

    image = ImageTk.PhotoImage(file=image_file)
    image_box.config(image=image)
    image_box.image = image

def main():
    idx = 0
    MTSD_classes = os.listdir(path=MTSD_FULLY_ANNOTATED_CLASSIFIED_CROPPED_IMAGES_TRAIN_DIR)

    root = tk.Tk()

    frame = tk.Frame(root)
    frame.pack()

    button = tk.Button(frame, text="QUIT", fg="RED",
                       command=lambda: imshow(image_dir=MTSD_FULLY_ANNOTATED_CLASSIFIED_CROPPED_IMAGES_TRAIN_DIR,
                                              MTSD_classes=MTSD_classes, idx=idx))
    button.pack(side=tk.LEFT)



    image_box = tk.Label(root)
    image_box.pack()

    # for cls in MTSD_classes:
    #     cls_img_dir = MTSD_FULLY_ANNOTATED_CLASSIFIED_CROPPED_IMAGES_TRAIN_DIR + cls + '/'
    #     img_name = os.listdir(path=cls_img_dir)
    #     print(img_name)
    root.mainloop()
    return

if __name__ == "__main__":
    main()