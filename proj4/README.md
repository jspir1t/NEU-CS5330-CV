# Project 4: Calibration and Augmented Reality

# Links
[Doc](https://wiki.khoury.northeastern.edu/x/XM15Bg)  
[Video](https://youtu.be/bvM--huCEiU)

## Environment
OS: Windows
IDE: CLion + mingw

## Instructions
First, you should have your folder configured like this:
```shell
│  CMakeLists.txt
│
├─img
│  ├─chessboard
│  └─circlesgrid
├─include
│      utils.h
│
├─objs
│      humanoid.obj
│      teapot.obj
│      teddy.obj
│
└─src
        ar.cpp
        feature.cpp
        main.cpp
        utils.cpp
```
### Build
Type in
```
cmake -DCMAKE_BUILD_TYPE=Release -G "CodeBlocks - MinGW Makefiles" -S . -B .\build
cmake --build .\build\
```
You will have a build folder, go to that folder.

### Calibration
The first parameter specify target, you could choose **chessboard** or **circlesgrid**
1. Chessboard
Type in this command, aim to the chessboard.
```shell
.\main.exe chessboard
```
2. Circles Grid 
Type in this command, aim to the circles grid
```shell
.\main.exe grid
```
Once there are corners drawn, press **'s'** to save the frame to corresponding folder under the img folder.  
After at least 5 frames are collected, press **'c'** to do the calibration. There will be a csv created after that in the parent folder.

### AR
The first parameter specify target, stick to your choice in the calibration
The second parameter specify the object you want to display, the options are **teapot**, **teddy** and **humanoid** in objs folder.
1. Chessboard
```shell
.\ar.exe chessboard teapot
```
2. Circles Grid
```shell
.\ar.exe circlesgrid teapot
```
There will be axes drawn in the chessboard.  
press **'w'** to show the 3D object and axes.  
Press **'a'** to show the axes only.

### Harris Corner && SIFT
```shell
.\feature.exe
```
It shows the Harris corners as default.  
Press **'f'** to show the SIFT corners.  
Press **'h'** to show the Harris corners.  

### Quit
Press 'q'