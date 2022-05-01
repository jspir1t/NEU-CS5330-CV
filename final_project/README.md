# A More Robust Model For QR Code Detection

#### Group Members
Jingtong Zhang, Yuqin Luo, Ying Bi

## Project Description
The goal of our project is the self-implemented detection and segmentation of QR codes based on the features of position
boxes, it should be invariant to translation, rotation, and scale. Also, we will conduct the experiment with other
open-source implementations like OpenCV and WeChat, then compare these results and analyze them.

## Instructions
### Build
```shell
cmake -DCMAKE_BUILD_TYPE=Release -G "CodeBlocks - MinGW Makefiles" -S . -B .\build
cmake --build ./build
```
### Execution
There are three modes for each method: **Real-time Video**, **Single Image**, and **Dataset Evaluation**

In the first two modes, you could press 's' to save the result to "results" folder, or 'q' to quit the execution. If
the qrcode is detected successfully, there will be green bounding box around it.

In the Dataset Evaluation mode, the results will be stored in the results/<mode_name>_eval folder, and a csv file named
<mode_name>.csv will be created in the "results" folder.

- Our Method(In "build" folder)
  - real-time video  
  ```.\main.exe video```
  - single image detection  
  The image name should be in the one in the "images" folder, such as qrcode, occlusion3...  
  ```.\main.exe image <image_name>```
  - dataset evaluation  
  ```.\main.exe eval```

- OpenCV(In "src" folder)
  - real-time video  
    ```python main.py opencv video```
  - single image detection  
    The image name should be in the one in the "images" folder, such as qrcode, occlusion3...  
    ```python main.py opencv image <image_name>```
  - dataset evaluation  
    ```python main.py opencv eval```

- WeChat(In "src" folder)
  - real-time video  
    ```python main.py wechat video```
  - single image detection  
    The image name should be in the one in the "images" folder, such as qrcode, occlusion3...  
    ```python main.py wechat image <image_name>```
  - dataset evaluation  
    ```python main.py wechat eval```

- Analyze  
  After all the evaluation for these three methods, run ```python main.py analyze``` to see the accuracy for each method
  like this:
  ```json
  {'far1': '0', 'far2': '0', 'far3': '0', 'far4': '0', 'far5': '0', 'far6': '0', 'lowlight1': '0', 'lowlight2': '0', 'lowlight3': '0', 'lowlight4': '0', 'lowlight5': '0', 'lowlight6': '0',
   'occlusion1': '0', 'occlusion2': '0', 'occlusion3': '0', 'occlusion4': '1', 'occlusion5': '0', 'occlusion6': '0', 'qrcode': '1', 'qrcode1': '1', 'qrcode2': '1', 'qrcode3': '0'}
  Accuracy for opencv: 0.18181818181818182
  {'far1': '1', 'far2': '0', 'far3': '1', 'far4': '1', 'far5': '0', 'far6': '1', 'lowlight1': '0', 'lowlight2': '1', 'lowlight3': '1', 'lowlight4': '0', 'lowlight5': '0', 'lowlight6': '0',
   'occlusion1': '0', 'occlusion2': '1', 'occlusion3': '0', 'occlusion4': '0', 'occlusion5': '1', 'occlusion6': '1', 'qrcode': '1', 'qrcode1': '1', 'qrcode2': '1', 'qrcode3': '1'}
  Accuracy for self: 0.5909090909090909
  {'far1': '1', 'far2': '1', 'far3': '1', 'far4': '1', 'far5': '1', 'far6': '1', 'lowlight1': '1', 'lowlight2': '1', 'lowlight3': '1', 'lowlight4': '1', 'lowlight5': '1', 'lowlight6': '0',
   'occlusion1': '0', 'occlusion2': '0', 'occlusion3': '1', 'occlusion4': '1', 'occlusion5': '0', 'occlusion6': '1', 'qrcode': '1', 'qrcode1': '0', 'qrcode2': '1', 'qrcode3': '1'}
  Accuracy for wechat: 0.7727272727272727
  ```


## Presentation URL

## Demo URL