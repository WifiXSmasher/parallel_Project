# parallel_Project
To Run the project , the following things need to be installed :
  1. YOLO models (s,m,x) [in the same dir as the codes]
  2. A labeled dataset [we used :https://www.kaggle.com/datasets/dtrnngc/ua-detrac-dataset]

To run the project (for generating all the graphs and comparing all the approaches):
  run the following commands :
  1. source venv/bin/activate [to activate the virtual env]
  2. pyhton3 run_all_experiments.py

a graph will be generated automatically 

COMMAND TO GENERATE THE VIDEO USING THE DATASET : ffmpeg -framerate 25 -i img%05d.jpg -c:v libx264 -preset medium -crf 23 -g 30 -sc_threshold 0 -bf 2 -pix_fmt yuv420p test_video.mp4

HERE -g 30 is for GOP of 30

DATASET USED :https://www.kaggle.com/datasets/dtrnngc/ua-detrac-dataset

NOTE : CUDA AND NVIDIA DRIVERS NEEDED 

