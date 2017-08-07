# 2D-DCGAN

## Requirements
- python3
- numpy 1.13+
- tensorflow 1.2+
- scipy 0.19+

## Usages
### Download dataset 
Download dataset first. You can choose `celebA` or `flowers` to download and convert to `*.npy` file, or `*.tfrecords` format.
If you use `*.tfrecords`, check [here](https://github.com/nmhkahn/DCGAN-tensorflow-slim.git) to see more information how to use.
Here using `*.npy` as model input format.
```shell
$ cd dataset
$ python download_and_convert.py -D celebA -F npy
```

The instructure above resizes `celebA` from 128 x 128 to 64 x 64, and saves it in `*.npy` file.

### Training 2D-DCGAN
Move to `2d` folder. Change your dataset location in `05-DCGAN_celeba.py`:
```python
-- 321 celeba = np.load('../dataset/celeba/celeba_64.npy')
++ 321 celeba = np.load('../dataset/celeba_64.npy')
```

Save and train the model:
```python
$ python 05-DCGAN_celeba.py
```

### Testing phase:
The training checkpoints are saved in `models`. Choose one checkpoint to test.
For example,
```python
$ python 05-DCGAN_celeba.py --notrain --ckpath models/celeba_64.ckpt-100
```
