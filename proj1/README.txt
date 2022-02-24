1. Report URL
https://wiki.khoury.northeastern.edu/x/vQ8dBg

2. Hardware
Operating System: Windows 10
IDE: Clion

3. Instructions for building and running the two executable files
Before all, please put all the files like below:

│  CMakeLists.txt
│
├─img
│      test.png
│
├─include
│      filter.h
│
└─src
        filter.cpp
        imgDisplay.cpp
        vidDisplay.cpp

Then go to the project path and run the commands below:
cmake -DCMAKE_BUILD_TYPE=Release -G "CodeBlocks - MinGW Makefiles" -S . -B .\build
cmake --build .\build\ --target image_display -- -j 6
cmake --build .\build\ --target video_display -- -j 6

After that, you should have the directory structure as:
├─build
│   ...
│
├─img
│      test.png
│
├─include
│      filter.h
│
└─src
        filter.cpp
        imgDisplay.cpp
        vidDisplay.cpp

Simply run the executable files by:
cd .\build
.\image_display.exe
.\video_display.exe

4. Instructions for testing extensions
Press space bar to show the origin image
Press 'n' to show the negative mat of itself(task 10)
Press '<' to decrease the contrast
Press '>' to increase the contrast
Press '-' to decrease the brightness
Press '+' to increase the brightness
Press 'p' to show the sepia effect
Press 'e' to show the emboss effect
The keypress for running the other tasks are identical to the project requirement