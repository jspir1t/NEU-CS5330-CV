## Report URL  
   https://wiki.khoury.northeastern.edu/x/1YV5Bg

## Hardware  
   Operating System: Windows 10  
   IDE: Clion

## Instructions
Before all, please put all the files like below:
```shell
│CMakeLists.txt
│
├─olympus
│      images...
│
├─include
│      features.h
│
└─src
       features.cpp
       main.cpp
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
├─olympus
│      images...
│
├─include
│      filter.h
│
└─src
      filter.cpp
      imgDisplay.cpp
      vidDisplay.cpp
```


Simply run the executable files by:
```shell
cd .\build
.\main.exe ..\olympus <task-number>
```
where task-number should be a number from 1 to 6, which stands for task 1 to task 6 (extension as task 6).  
if you want to see the effect of blue bins in task 4, change the definition in features.h:
```shell
#define CUSTOM_HIST_IMAGE "..\\olympus\\pic.0287.jpg"
```