<!-- # project_template  -->


# THE FAIRNESS STITCH: Unveiling the Potential of Model Stitching in Neural Network De-Biasing
Implementation of the paper "THE FAIRNESS STITCH: Unveiling the Potential of Model Stitching in Neural Network De-Biasing", by Modar Sulaiman and Kallol Roy.

This paper introduces the framework "The Fairness Stitch (TFS)" to enhance fairness in deep learning models. It combines model stitching and training using with fairness constraints. 




## Datasets
The CelebA and UTKFace datasets used our experiments. 

[CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html): It is a collection of celebrity faces, comprising more than 200,000 images annotated with 40 distinct attributes, including facial landmarks, gender, age, hair color, glasses, etc. 

[UTKFace dataset](https://susanqq.github.io/UTKFace/): It is a publicly accessible face dataset boasting an extensive age range from newborns to individuals aged up to 116 years old, was employed in our study.



## Installation

* Create the environment and install the required libaries

* This code depends on the following packages:

 1. `Torch`
 2. `NumPy`
 3. `pillow`
 4. `fairlearn`

* To install all the required dependencies, run the following command:
```sh
pip install -r requirements.txt
```


## Experiments

 - We provide code for comparing [FDR](https://arxiv.org/pdf/2304.03935.pdf) with our framework 'TFS'. 

 - `src/get_data.py` is used for downloading the CelebA dataset. However, you can download the datasets from the original sources.

 - The `main.py` file contains code for fine-tuning FDR and training our framework 'TFS'.

 - The `testing.py` file is used for testing FDR and TFS,

Please run first
```sh
python main.py

```


### License
The source code for the site is licensed under the MIT license, which you can find in the MIT-LICENSE.txt file.



