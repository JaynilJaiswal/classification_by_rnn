# Classification_by_rnn
In this project I have implemented a simple name classifier which classifies the names into their possible origin country.I 
have used one-layered RNN taking character by character input.
## Run
- First run `train.py` file using command `python train.py` in terminal given that python v3.0 is installed in your system.
- Next run `predict.py` file using command `python predict.py`.This will give us the % accuracy of our classifier on the given
  dataset.
## Dataset used
The dataset used is available in this repository.It contains of more than 20000 names with their origins.
## Model
The model used for this project is taken from [this](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) tutorial.
Each character is being converted into a one-hot-vector and feeded to layer along with the output of the previous character output.
The RNN learns by processing each character while using the information from the previous characters as well. 
<p align="center">
<img src="https://i.imgur.com/Z2xbySO.png" /></div>

## Code reference
This code is being refered from [this tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) and reformed it according to our needs.

## Results and Observations
- By default the dataset is divided into 75% training and 25% test data with 10 epochs .It gave an accuracy performance of approximately 74.75%.
- When I changed the dataset training:test ratio to 90:10 it dropped to 72.06%.
- When no. of epochs is gradually increased from 1 to 10 at first the accuracy increases significantly like about 2 to 3 % but after a while it doesn't increase much.
