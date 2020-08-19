# Spike

1. Separate out trending and stationary data
    - For each X bars e.g. 100 bars 
    - Trending if price diff from start to end > ATR of past 500 bars inclusive of X bars and > round trip commission and spread 
    - Else stationary 
2. GAN for trending and GAN for stationary data
    - Train 2 GANS, 1 for trending data and 1 for stationary data
    - Discriminator sees masked input of only first Y bars where Y < X e.g 50 bars
3. With new input, compare the outputs of discriminators for trending and stationary data. 
    - For every new X bars, 
    - Argmax the output of the 2 discriminators to predict for trending or stationary 



