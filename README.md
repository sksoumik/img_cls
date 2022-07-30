#### Install Dependencies

```
pip install -r requirements.txt
```

#### Dataset link for training the ML model

[kaggle - Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)4

This Data contains around 25k images of size 150x150 distributed under 6 categories.

```
{'buildings' -> 0,
'forest' -> 1,
'glacier' -> 2,
'mountain' -> 3,
'sea' -> 4,
'street' -> 5 }
```

Create a directory called `dataset/` in the root folder.
Download the dataset from kaggle and place it inside the dataset directory. Unzip the data.

#### Train ML model

Dependencies:

```
pip install tf-nightly-gpu

```

Train the model

```
python train.py
```

Optional arguments can be passed if the data paths are different.

```
python train.py -train [train images data path] -valid [valid images data path] -pred [test/prediction images data path]
```

This will create a tflite model using MobileNetV2 and save the model in the `static/` directory.

#### Run the app locally

```
uvicorn main:app
```

