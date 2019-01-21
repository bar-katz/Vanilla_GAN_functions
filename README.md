# GAN for generating points of learned function

This project was made in order to practice GANs.

``gan_model.py`` uses a Generator and Discriminator architecture to learn a function from sample points.

``generate_data.py`` create .pkl files containing samples of 3 different functions.


## Usage

1. Run ``python generate_data.py`` to create real data files

2. Then Run
    ```sh
    python gan_model.py {real_data_file}
    ```

## Results

f(x) = x<br>
![line_pred](https://user-images.githubusercontent.com/33622626/51479649-8929dc80-1d97-11e9-9bcb-404f39444b3f.png)

f(x) = x^2<br>
![par_pred](https://user-images.githubusercontent.com/33622626/51479669-99da5280-1d97-11e9-8b07-fe5c04e4de0d.png)

spiral function(see ``generate_data.py``)<br>
![spiral_pred](https://user-images.githubusercontent.com/33622626/51479685-a5c61480-1d97-11e9-849a-6afb1258ec84.png)

## Built With
* [PyTorch](https://pytorch.org/docs/stable/index.html) – a deep learning platform


## Author

Bar Katz – [bar-katz on github](https://github.com/bar-katz) – barkatz138@gmail.com
