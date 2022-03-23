## Report URL
https://wiki.khoury.northeastern.edu/x/gqd5Bg

## Hardware
Operating System: Windows 10  
IDE: Clion

## Instructions
Before all, please put all the files like below:
```shell
│CMakeLists.txt
│
│
├─include
│      classifier.h
│      retrieval.h
│
└─src
       classifier.cpp
       client.cpp
       retrieval.cpp
```


Then go to the project path and run the commands below:
```shell
cmake -DCMAKE_BUILD_TYPE=Release -G "CodeBlocks - MinGW Makefiles" -S . -B .\build
cmake --build .\build\ --target main -- -j 6
```


After that, you should have the directory structure as:
```shell
|CMakeLists.txt
|
├─build
│   ...
│
├─include
│      classifier.h
│      retrieval.h
│
└─src
       classifier.cpp
       client.cpp
       retrieval.cpp
```


Simply run the executable files by:
```shell
cd .\build
.\main.exe
```

## How to play
There will always be five windows displaying the image in different phases, which are original image,
thresholded image, cleaned-up image, segmentation image and classification image.  

There are three database files involved in this project, which are "features.txt" for storing training data features,
"test_features.txt" for storing test data features and "../evaluation.txt" for storing the confusion matrix for the test dataset.

Once launching the client, you are in the segmentation mode(default), where the segmentation window will
show the components retrieved and colored. Moreover, it will mark the components with bounding box,
centroid and least central axis by green line.

- Press "q" will exit the program  
- Press "a' will save the four windows into origin.jpg, threshold.jpg, cleanup.jpg, components.jpg  
- Press " " will enter segmentation mode(default), where the segmentation window will show all the top n components after filtering(bigger than min_area, not adjacent to border)
- Press "t" will enter training mode, the segmentation window will only mark the major component for you to save the feature to test dataset
- Press "p" will enter the evaluation-preparation mode, it shows the same thing as training mode except that the feature would be saved to test dataset
- Press "s" in training mode or evaluation-preparation mode will prompt the stdin in terminal for you to type in the name of the current feature
- Press "n" will trigger the nearest neighbor classifier for all the components in the segmentation window
- Press "k" will trigger the knn classifier for all the components in the segmentation window
- Press "e" will trigger the evaluation for the test dataset based on the training dataset

#### Tips
1. The segmentation window may show "No component detected", which means there is no component detected after the clean
and filter mentioned above. No worry, it is normal especially when you put nothing on the white paper.
2. For knn classifier and evaluation with knn, if the number of each class in the training dataset is less than k, it will
prevent your operation since it obeys the requirement of knn.
3. The "s" key will only work in training mode and evaluation-preparation mode, it will save the feature marked to
corresponding dataset.

Enjoy the client :)