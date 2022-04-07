# Project 5: Recognition using Deep Networks

# Links
[Doc](https://wiki.khoury.northeastern.edu/x/0OB5Bg)

## Environment
OS: Windows
IDE: Pycharm + Python3.8

## Instructions
First, you should have your folder configured like this:
```shell
├─data                               
│
├─extensions
│      car.jpg
│      VGG16_exploration.py
│      gabor.py
│      live_video_digit_recognition.py
│
├─task1
│  │  diagram
│  │  diagram.png
│  │  model.py
│  │  predict.py
│  │
│  ├─results
│  │
│  ├─test_digits
│        0.png
│        1.png
│        2.png
│        3.png
│        4.png
│        5.png
│        6.png
│        7.png
│        8.png
│        9.png
│  
│
├─task2
│      examine.py
│
├─task3
│  │  embedding.py
│  │
│  ├─greek
│  │      alpha_001.png
│  │      alpha_002.png
│  │      alpha_003.png
│  │      alpha_004.png
│  │      alpha_005.png
│  │      alpha_006.png
│  │      alpha_007.png
│  │      alpha_008.png
│  │      alpha_009.png
│  │      beta_001.png
│  │      beta_002.png
│  │      beta_003.png
│  │      beta_004.png
│  │      beta_005.png
│  │      gamma_004.png
│  │      gamma_005.png
│  │      gamma_006.png
│  │      gamma_007.png
│  │      gamma_008.png
│  │      gamma_009.png
│  │
│  └─test_greek
│          alpha_1.png
│          alpha_2.png
│          beta_1.png
│          beta_2.png
│          gamma_1.png
│          gamma_2.png
│
├─task4
   │  experiments.ipynb
   │  task4.pdf
   │
   └─result
```

For each .py file, simply run python <name>.py to run the file.

### Task-1
run ```python  python model.py``` to download the MNIST dataset to the ../data folder, train the model and save the 
results into task1/results folder.  
Also, you have to install torchviz and define Graphviz in your system path to get the diagram. If you do not want, 
please comment the lines from 147 to 152 in model.py
run ```python  python predict.py``` to make the prediction on first example in test dataset and the new input in test_digits.

### Task-2
run ```python  python examine.py``` to generate those plots for first layer, and it will also create a truncated model.

### Task-3
run ```python  python embedding.py``` to generate the csv files, print out the SSD result and make prediction on the new
input in test_greek folder.

### Task-4
Please look at the task4.pdf if you cannot open the ipynb notebook. If you want to run the code, please open the ipynb
file to run the cells manually. It will save the results in the "result" folder under "task4" folder.

### Extensions
There are three python files under the "extensions" folder, run whatever you want by ```python python <name>.py```, it
will generate the plots or standard output in the command line.  

Hope you enjoy the execution :)