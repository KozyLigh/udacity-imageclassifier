import numpy as np
import json

import argparse

import projectUtils

#Command Line Arguments

ap = argparse.ArgumentParser(
    description='predict-file')
ap.add_argument('input_img', default='./flowers/test/1/image_06752.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='./checkpoint.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

# parse args and set to parameters
pa = ap.parse_args()
image_path = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img
path = pa.checkpoint


print("#### args start ####")
print(image_path)
print(number_of_outputs)
print(power)
print(input_img)
print(path)
print("#### args end ####")

dataloaders, dataloaders_validation, dataloaders_test, image_datasets = projectUtils.load_data()

model = projectUtils.load_model(path)

with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)

probabilities = projectUtils.predict(image_path, model, number_of_outputs, power)

labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0].cpu())]
probability = np.array(probabilities[0][0].cpu())

i=0
while i < number_of_outputs:
    print("{} with probability {}".format(labels[i], probability[i]))
    i += 1

print("#### Prediction is finished ####")