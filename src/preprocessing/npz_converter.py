import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pandas as pd

"""
# Found in https://docs.deepcognition.ai/how-to-guides/datasets/create-numpy-dataset-from-images,
# but lightly modified according to my requirements
"""

image_dataset_dir = "G:\\My Drive\\Colab Notebooks\\COMP 432\\Project\\data"
new_dataset_folder = "G:\\My Drive\\Colab Notebooks\\COMP 432\\Project\\npz"

dataset = {
    "image": [],
    "label": []
}

for label in os.listdir(image_dataset_dir):
    images_dir = image_dataset_dir + "/" + label
    if not os.path.isdir(images_dir):
        continue
    for image_file in os.listdir(images_dir):
        if not image_file.endswith((".jpg", ".png", ".tiff", ".bmp")):
            continue

        #  Get image
        img = load_img(os.path.join(image_dataset_dir, label, image_file))
        x = img_to_array(img)


        # rel_path = label + "/" + image_file
        # os.makedirs(new_dataset_folder + "/" + label, exist_ok=True)
        # npz_file = os.path.join(new_dataset_folder, rel_path)
        #        np.savez(npz_file, x)
        dataset["image"].append(x)
        dataset["label"].append(label)

#  df = pd.DataFrame(dataset)
#  df.to_csv(os.path.join(new_dataset_folder, "train.csv"), index=False)
np.savez(new_dataset_folder + "\\" + 'data.npz', X=dataset["image"], y=dataset["label"])

print('Dataset converted to npz and saved here at %s ' % new_dataset_folder)

#  df.head()
