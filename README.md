# ICM-Final-Project
Use SVD to compress images then resize using given interpolation method.
```
python compress_res.py --mode {bilinear or bicubic} --input_image {input image path}
```
Scale up the input image with four different methods:

bilinear, bicubic, Lanczos, and EDSR.
```
python resize.py --input_image {input image path} --scale_factor {2.0, 3.0, 4.0}
```
## References
- [Pretrained Model](https://github.com/Saafke/EDSR_Tensorflow/tree/master)

- https://photock.jp/category/
