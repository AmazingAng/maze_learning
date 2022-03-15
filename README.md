# Exploration Maze
Hippocampal coding of maze exploration

# How to run the code?

1. Download behavCam1.avi in MiaoLab Server under folder H:\miniscope recording\maze\4_21_2020\10031H17_M15_S0 , and add to main folder to run the code.
2. Run cv_maze_raw_M10031_new.ipynb to process behavior data.
3. Run ms_decoding.ipynb to process calcium data.
4. Run ms_analysis.ipynb to decode behavior data from calcium data.
5. Run Behav_maze.py to process all behavior data based on an Excel file that records experimental details and file paths.
6. Run ms_maze.py to process all calcium data based on an Excel file that records experimental details and file paths.
7. Run Days_Aligned_online.ipynb to align data in different days.
8. Run IncorrectPath_ratemap.ipynb to draw the ratemap of incorrect path.
9. maze_graph.py and maze_utils.py are both necessary for the codes' running

# Data:
## Behavior results:

correct time, wrong time: time (seconds) spent on correct (both location and direction) and incorrect path

decision_rate: percentage of corrected decision made (计算方法可能有错)

half_level_proportion: proportion of time spent on first half levels of maze.

stop_time_mean, stop_time median: mean and median of stopping time (number of consecutive frames that mice remain on the same node).

speed (cm/s): mean of instantaneous speed (speed at each frame)

