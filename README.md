# LkGAN
A novel GAN that generalizes the LSGAN loss function. Confers better training stability and image quality.

# Requirements
To install requirements
```
pip install -r requirements.txt
```
Use Python version 3.7.4.

# Training
## Training LkGAN-k
To train LkGAN-k, run this command
```
python3 lkgan_static.py
```
which will prompt you to input k, version, trial number and seed. 
We used seed 123, 5005, 1600, 199621, 60677, 20435, 15859, 33764, 79878,
36123 for trials 1 to 10, respectively.
The version determines the parameters of &alpha;, &beta;, and &gamma;.
The table below details this information:

| Version | &alpha; | &beta; | &gamma; |
| :---: | :---: | :---: | :---: |
| 1 | 0.6 | 0.4 | 0.5 |
| 2 | 1 | 0 | 0.5 |
| 3 | 0 | 1 | 1 |

Note that if you input k = 2, you will run LSGAN. 

## Training LkGAN-[1, 3]
To train LkGAN-[1, 3], run this command
```
python3 lkgan_varying_k.py
```
which will prompt you to input the version, trial number and seed.
We used the same seeds as before for trials 1 to 10 and the 
version determines the the parameters of &alpha;, &beta;, and &gamma;.
Please refer to the previous table for more details.

The code will automatically calculate the FID scores during training and
will save the images for the epoch where the lowest FID score occurs as `predictions.npy`.

# Further information 
The weights of each layer were initialized using a Gaussian random variable with mean 0 and standard deviation
0.01.
We used the Adam optimizer with a learning rate of
 2 x 10<sup>-4</sup>, &beta;<sub>1</sub> = 0.5, &beta;<sub>2</sub> = 0.999, and
&epsilon; = 1 x 10<sup>-7</sup> for both networks.
The batch size was chosen to be 100 for the 60,000 MNIST images.
The total number of epochs was 100 for the MNIST images.
