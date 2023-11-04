import sys
import numpy as np
import torch
import os
import wget
import zipfile






##############


data_root = "datasets/celeba/"

base_url = "https://graal.ift.ulaval.ca/public/celeba/"

file_list = [
    "img_align_celeba.zip",
    "list_attr_celeba.txt",
    "identity_CelebA.txt",
    "list_bbox_celeba.txt",
    "list_landmarks_align_celeba.txt",
    "list_eval_partition.txt",
]

# Path to folder with the dataset
dataset_folder = f"{data_root}/celeba"
os.makedirs(dataset_folder, exist_ok=True)

for file in file_list:
    url = f"{base_url}/{file}"
    if not os.path.exists(f"{dataset_folder}/{file}"):
        wget.download(url, f"{dataset_folder}/{file}")

  
with zipfile.ZipFile("datasets/celeba/img_align_celeba.zip","r") as zip_ref:
    zip_ref.extractall("datasets/celeba/img_align_celeba")
