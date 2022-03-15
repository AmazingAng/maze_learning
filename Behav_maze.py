# 新建图像保存子目录，代码来源 https://www.php.cn/python-tutorials-424348.html
from scipy.io import loadmat # load matlab file
import matplotlib.pyplot as plt # plot
import pickle # save python data
import numpy as np 
import time
import os
import pdb
import scipy.stats
import networkx as nx
import cv2 # perspective transformation

from maze_graph import *
from maze_utils2 import *

# 创建路径的函数
def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print("     "+path + ' 创建成功')
        return True
    else:
        print("     "+path + ' 目录已存在')
        return False

# function to interpolate nodes in the maze
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def interpolate_pos_maze(behav_time, behav_nodes, behav_time_original, graph, test_maze):
    # interpolate behavior position on graph
    behav_nodes_interpolated = np.zeros_like(behav_time_original)
    # initialize first node to 1
    behav_nodes_interpolated[0] = 1
    before_node = 1
    after_node = 1

    # missing data from start
    if behav_time_original[0] < behav_time[0]:
        behav_nodes_interpolated[0:np.where(behav_time_original == behav_time[0])[0][0]] = before_node

    for i in range(len(behav_time)-1):
        before_time = behav_time[i]
        after_time = behav_time[i+1]
        before_node = behav_nodes[i]
        after_node = behav_nodes[i+1]
        before_index = np.where(behav_time_original == before_time)[0][0]
        after_index = np.where(behav_time_original == after_time)[0][0]
        behav_nodes_interpolated[before_index] = before_node
        behav_nodes_interpolated[after_index] = after_node
        # if there is missing value in between
        if after_index != before_index+1:
            # if nodes before and after missing frames are equal, assign to frames
            if before_node == after_node:
                behav_nodes_interpolated[(before_index+1):after_index] = before_node
            # when nodes are different, check if previous node is in the neighbor of after node
            elif before_node in graph[after_node]:
                # assign first half to previous node and latter half to after node
                half_index = before_index + (after_index-before_index)//2 +1
                behav_nodes_interpolated[(before_index+1):half_index] = before_node
                behav_nodes_interpolated[half_index:after_index] = after_node
            else:
                # the nodes are different and are not neighbor to each other, we need to find shortest path and interpolate
                path_nodes = test_maze.BFS_SP(graph = graph, start = before_node, goal = after_node)[1:-1]
                if len(path_nodes) == 1:
                    behav_nodes_interpolated[(before_index+1):after_index] = path_nodes[0]
                else:
                    # print([before_index, after_index])
                    tmp_list = behav_nodes_interpolated[(before_index+1):after_index]
                    k, m = divmod(len(tmp_list), len(path_nodes))
                    for i in range(len(path_nodes)):
                        tmp_list[i*k+min(i, m):(i+1)*k+min(i+1, m)] = path_nodes[i]
    # all frame should be interpolated
    #pdb.set_trace()
    assert(len(np.where(behav_nodes_interpolated == 0)[0]) == 0)
    return(behav_nodes_interpolated)

def plot_ratemap(ratemap, ax=None, title=None, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
    """Plot ratemaps."""
    if ax is None:
        ax = plt.gca()
    # Plot the ratemap
    ax.imshow(ratemap, interpolation='none', *args, **kwargs)
    # ax.pcolormesh(ratemap, *args, **kwargs)
    ax.axis('off')
    if title is not None:
        ax.set_title(title)
        
# calculate mean frame
def get_meanframe(video_name):
    cap = cv2.VideoCapture(video_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(length):    # Capture frame-by-frame
        ret, frame = cap.read()  # ret = 1 if the video is captured; frame is the image
        if i == 0: # initialize mean frame
            mean_frame = np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        # Our operations on the frame come here    
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/length
        # img = frame/length
        mean_frame = mean_frame + img
    
    return mean_frame
    # img = cv2.flip(frame,1)   # flip left-right  
    # img = cv2.flip(img,0)     # flip up-down
    # print(i)
    # Display the resulting image
    # cv2.imshow('Video Capture',img)
# When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	# maxWidth = max(int(widthA), int(widthB))
	maxWidth = 360
    # compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	# maxHeight = max(int(heightA), int(heightB))
	maxHeight = 360
    # now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped_image = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# warped_positions = cv2.perspectiveTransform(np.array([ori_positions]) , M)
	# return the warped image
	return warped_image, M

def transform_bin(bin_numbers):
    # rotate bin by 90 degree
    y = 13 - bin_numbers[1,:]
    x = bin_numbers[0,:]
    return np.array([x,y])

class PolygonDrawer(object):
    def __init__(self, window_name):
        self.window_name = window_name # Name for our window

        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon


    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every smouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("     Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("     Completing polygon with %d points." % len(self.points))
            self.done = True


    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, equ_meanframe)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            canvas = equ_meanframe
            # canvas = np.zeros(CANVAS_SIZE, np.uint8)
            cv2.polylines(canvas,  np.int32([ori_positions]), False, FINAL_LINE_COLOR, 1)

            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 3)
                # And  also show what the current segment would look like
                # cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing
        canvas = equ_meanframe

        # of a filled polygon
        if (len(self.points) > 0):
            cv2.polylines(canvas, np.array([self.points]),True, FINAL_LINE_COLOR, thickness = 5)
        # And show it
        cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()
        
        # Four points transform
        warped_image, M = four_point_transform(equ_meanframe, np.asarray(self.points))
        cv2.imshow("Processed Maze", warped_image)
        warped_positions = cv2.perspectiveTransform(np.array([ori_positions]) , M)[0]
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        cv2.destroyWindow("Processed Maze")
       
        return warped_image, warped_positions, M

def calculate_ratemap(agent_poss, activation, statistic='mean'):
    xs = agent_poss[:,0]
    ys = agent_poss[:,1]

    return scipy.stats.binned_statistic_2d(
        xs,
        ys,
        activation,
        bins=_nbins,
        statistic=statistic,
        range=_coords_range,
        expand_binnumbers = True)

# 主程序，作用于循环读取 |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
def Run_all_mice(mylist):
    # 主要参数赋值，参数来源Excel
    totalpath = "G:\YSY"
    date = mylist[0]
    NumOfMice = mylist[1]
    maze_type = mylist[2]
    row_correct = mylist[3]
    file_path = mylist[4]
    print(file_path)
    

    path = os.path.join(file_path,"behav_new.mat")
    behav_mat = loadmat(path); # read data
    
    mkpath = [os.path.join(totalpath),
    os.path.join(totalpath,NumOfMice),
    os.path.join(totalpath,NumOfMice,date),
    os.path.join(totalpath,NumOfMice,date,"behav"),
    os.path.join(totalpath,NumOfMice,date,"neural_activity"),
    os.path.join(totalpath,NumOfMice,date,"decoding"),
    os.path.join(totalpath, NumOfMice, "Days_aligned"),
    os.path.join(totalpath, NumOfMice, "Days_aligned", "maze"+str(maze_type))]

    for i in range(len(mkpath)):
        mkdir(mkpath[i])
    
    # --------------------------------------------------------------------------------------------------------------------------------
    # 矫正开始、结束数据
    import pandas as pd
    correct_time_data = pd.read_excel("G:\YSY\mice_maze_metadata_time_correction.xlsx", sheet_name = "training_recording_new")

    # position number 13
    behav_positions = behav_mat['behav']["position_original"][0][0]
    behav_time = behav_mat['behav']['time'][0][0]
    length = int(len(behav_positions))

    #手动设置删除帧数，存在“mice_maze_metadata_time_correction.xlsx”中
    delete_start = int(correct_time_data['start_time'][row_correct])
    delete_end = int(correct_time_data['end_time'][row_correct])

    behav_positions = np.delete(behav_positions, range(delete_end,length), axis = 0)
    behav_positions = np.delete(behav_positions, range(0,delete_start), axis = 0)
    behav_time = np.delete(behav_time, range(delete_end,length))
    behav_time = np.delete(behav_time, range(0,delete_start))
    # delete first 50 frames (~1.7s), when the mouse is not in the maze
    # should replace to exact starting time later!
    #     delete_start = 50
    #     behav_positions = np.delete(behav_positions, range(0,delete_start),0)
    #     behav_time = np.delete(behav_time, range(0,delete_start))

    # delete NAN values
    nan_mask = np.isnan(behav_positions).any(axis=1)
    behav_time_original = behav_time
    behav_positions = behav_positions[~nan_mask, ]
    behav_time = behav_time[~nan_mask]
    plt.plot(behav_positions[:,0], behav_positions[:,1])

    path = os.path.join(totalpath,NumOfMice,date,"behav",'Trace_without_smooth_Raw.png')
    plt.savefig(path)
    print("     Figure 1 'Trace_without_smooth_Raw' is done...")
    plt.close()
    
    # data cleaning: delete wrong data point by filtering speed ------------------------------------------------------------------------
    behav_len = behav_positions.shape[0]
    nan_seq_len = 0
    delta = 0
    max_speed = 60 # 100 cm/s

    behav_positions_tmp = behav_positions
    behav_time_tmp = behav_time
    good_ind_diff = 1
    for i in range(1,behav_len):
        delta = np.sqrt(np.sum(np.square(behav_positions[i,] - behav_positions[i-good_ind_diff,])))
        time_diff = behav_time[i] - behav_time[i-good_ind_diff]
        speed_tmp = delta/time_diff*1000
        if speed_tmp > max_speed:
            # print(i, behav_positions[i,],behav_positions[i-1,], speed_tmp, time_diff, delta)
            behav_positions_tmp[i,] = np.nan
            good_ind_diff += 1
        else:
            good_ind_diff = 1

    nan_mask2 = np.isnan(behav_positions_tmp).any(axis=1)
    behav_positions = behav_positions_tmp[~nan_mask2, ]
    behav_time = behav_time_tmp[~nan_mask2]
    
    # get start, end time --------------------------------------------------------------------------------------------------------------
    start_time = behav_time[0]
    end_time = behav_time[-1]
    total_time = end_time- start_time
    stay_time = np.append(np.ediff1d(behav_time),33)
    # get end index
    end_index = np.where(behav_time_original == end_time)[0][0]
    behav_time_original = behav_time_original[0:(end_index+1)]
    # define behavior x, y, roi
    behav_x = behav_positions[:,0]
    behav_y = behav_positions[:,1]
    behav_roi = behav_mat['behav']['ROI'][0][0][0]
    track_length = behav_mat['behav']['trackLength'][0][0][0]

    # plt.plot(behav_positions[:,0], behav_positions[:,1])
    plt.plot(behav_positions[:,0], behav_positions[:,1])

    path = os.path.join(totalpath,NumOfMice,date,"behav",'Trace_without_smooth.png')
    plt.savefig(path)
    print("     Figure 2 'Trace_without_smooth' is done...")
    plt.close()
    
    # plot trace on mean_frame

    # convert ROI position to frame position
    ori_positions = behav_positions * behav_roi[2]/track_length +  [behav_roi[0], behav_roi[1]]

    plt.plot(ori_positions[:,0],ori_positions[:,1], color = "red")
    plt.imshow(mean_frame.astype(int))

    path = os.path.join(totalpath,NumOfMice,date,"behav",'TraceOnMeanFrame_Raw.png')
    plt.savefig(path)
    plt.close()
    print("     Figure 3 'TraceOnMeanFrame_Raw' is done...")
    print("     Click in the window...")   
    # ===============================================================================================================================
    # ori_positions = behav_positions * behav_roi[2]/track_length + 
    FINAL_LINE_COLOR = (255, 100, 0)
    WORKING_LINE_COLOR = (127, 127, 127)

    # =================================================================================================================================
    roi = np.int32([behav_roi[0], behav_roi[0]+behav_roi[2], behav_roi[1], behav_roi[1]+behav_roi[3]])
    if __name__ == "__main__":
        pd = PolygonDrawer("Original: select 4 maze corners")
        warped_image,warped_positions, M  = pd.run()
        cv2.imwrite("polygon.png", warped_image)
        print("     Polygon = %s" % pd.points)
        
        path = os.path.join(totalpath, NumOfMice, date, "PerspTrans.pkl")
        with open(path, 'wb') as f:
            pickle.dump(M, f)
        
    cv2.destroyWindow("Original: select 4 maze corners")
    cv2.destroyWindow("Processed Maze")
    print("     The window is closed.")
    
    maxWidth = 360
    maxHeight = 360
    neg_mask = (warped_positions<0).any(axis =1)
    processed_pos = warped_positions
    processed_pos[warped_positions <0] = 0
    processed_pos[warped_positions[:,0] >maxWidth, 0] = maxWidth
    processed_pos[warped_positions[:,1] >maxHeight, 1] = maxHeight

    plt.imshow(warped_image)
    plt.plot(processed_pos[:,0], processed_pos[:,1], color = "red")

    path = os.path.join(totalpath,NumOfMice,date,"behav",'TraceOnMeanFrame.png')
    plt.savefig(path)
    path = os.path.join(totalpath,NumOfMice,date,"behav",'TraceOnMeanFrame.pdf')
    plt.savefig(path)
    plt.close()
    print("     Figure 4 'TraceOnMeanFrame' is done...")
    
    _nbins = 12
    _coords_range = [[0,maxWidth +0.01],[0, maxHeight+0.01]]
    activation = stay_time
    occu_time, xbin_edges, ybin_edges, bin_numbers = calculate_ratemap(processed_pos, activation, statistic = "sum")
    
    # plot rate map ---------------------------------------------------------------------------------------------------------------------------------
    # Plot the activation maps
    #plot_ratemap(np.transpose(occu_time), cmap=cm, alpha = 0.8)
    plt.imshow(warped_image)
    plt.scatter(processed_pos[:,0], processed_pos[:,1], color = "red")

    ax = plt.gca()
    ticks = np.linspace(0, 360, _nbins+1)
    ax.pcolor(ticks, ticks, np.transpose(occu_time), cmap='hot', alpha = 0.6)

    # plt.scatter([0,1, 2, 3], [0,1, 2, 3])
    path = os.path.join(totalpath,NumOfMice,date,"behav",'warped_image.png')
    plt.savefig(path)
    path = os.path.join(totalpath,NumOfMice,date,"behav",'warped_image.pdf')
    plt.savefig(path)  
    print("     Figure 5 'warped_image' is done...")
    plt.close()
    cm="hot"
    plot_ratemap(np.transpose(occu_time), cmap=cm, alpha = 0.8)

    path = os.path.join(totalpath,NumOfMice,date,"behav",'rate_map.png')
    plt.savefig(path)
    path = os.path.join(totalpath,NumOfMice,date,"behav",'rate_map.pdf')
    plt.savefig(path)
    print("     !!!  Figure 6 'rate_map' is done...")  
    plt.close()
    
    # Maze to graph (stored in list!)--------------------------------------------------------------------------------------------------------------------------
    # 12 x 12 maze, 144 cells
    # Cell order: left -> right, up -> down
    transformed_bin_number = transform_bin(bin_numbers)
    nx = 12
    ny = 12
    graph = maze1_graph if maze_type ==1 else maze2_graph
    
    start_node = 1
    end_node = 144
    test_maze = Maze(nx, ny, graph)
    test_maze.make_maze()
    shortest_path = BFS_SP(graph, start_node, end_node)
    cell_dists = test_maze.cell_dists
    behav_nodes = test_maze.loc_to_idx(transformed_bin_number[0,:]-1, transformed_bin_number[1,:]-1)
    
    behav_nodes_interpolated = interpolate_pos_maze(behav_time, behav_nodes, behav_time_original, graph, test_maze)
    # output: behav_nodes_interpolated ------------------------------------------------------------------------------------------------------------
    def get_direction(behav_nodes_interpolated, test_maze, shortest_path):
    # Function to get direction (right or wrong)
    # 1 is correct direction (to goal on correct path, or to correct path on incorrect path)
    # 0 is incorrect direction (to start on correct path, or to leaf on incorrect path)
        behav_dir = np.zeros_like(behav_nodes_interpolated)
        behav_dir[0] = 1
        behav_dir[-1] = 1
        for i in range(1, len(behav_dir)-1):
            # if current node equals next node, direction does not change
            if behav_nodes_interpolated[i] == behav_nodes_interpolated[i+1]:
                behav_dir[i] = behav_dir[i-1]
            else:
                if cell_dists[behav_nodes_interpolated[i]-1] > cell_dists[behav_nodes_interpolated[i+1]-1]:
                    behav_dir[i] = 1
                else:
                    behav_dir[i] = 0
        return(behav_dir)
    behav_dir = get_direction(behav_nodes_interpolated, test_maze, shortest_path)
    stay_time_original = np.append(np.ediff1d(behav_time_original),0)

    # calculate time spent in each node
    correct_time = 0    
    wrong_time = 0
    for i in range(len(stay_time_original)):
        if (behav_nodes_interpolated[i] in shortest_path) & (behav_dir[i] == 1):
            correct_time += stay_time_original[i]
        else:
            wrong_time += stay_time_original[i]
    print("     Total time: " + str((correct_time+wrong_time)/1000) + " s, \n Time on correct path & direction: " + str(correct_time/1000) + " s (" + "{:.2f}".format(correct_time/total_time*100) + "%), \n Time on incorrect path or direction : "+ str(wrong_time/1000) + " s (" + "{:.2f}".format(wrong_time/total_time*100) + "%)",end='\n\n')

    # correct decision rate: current frame at decision point, next frame at next cell in among the shortest path--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # total_decision = np.zeros(len(test_maze.decision_nodes))
    correct_decision = np.zeros(len(test_maze.decision_nodes))
    incorrect_decision = np.zeros(len(test_maze.decision_nodes))
    for i in range(len(behav_nodes_interpolated)-1):
        # check current node is a decision node, and not the same as next node (decision is made)
        if (behav_nodes_interpolated[i] in test_maze.decision_nodes) & (behav_nodes_interpolated[i+1] in graph.get(behav_nodes_interpolated[i])):
            # decision node index
            current_decision_node = behav_nodes_interpolated[i]
            next_node = behav_nodes_interpolated[i+1]
            current_idx_decision = test_maze.decision_nodes.index(current_decision_node)
            if cell_dists[current_decision_node - 1] >  cell_dists[next_node]:
                correct_decision[current_idx_decision] += 1
            else:
                incorrect_decision[current_idx_decision] += 1
        
        
    total_decision = correct_decision + incorrect_decision
    # correct decision rate for each decision node
    # correct_decision/(total_decision + 1e-12)

    # overall correct rate
    decision_rate = sum(correct_decision)/sum(total_decision)
    print("     Overall Correct Rate: " + "{:.2f}".format(decision_rate*100)+ "%",end='\n\n')
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # time spent in each level
    total_levels = len(test_maze.decision_nodes)
    time_levels = np.zeros(total_levels)
    for i in range(len(stay_time_original)):
        level_ind = test_maze.cell_levels[behav_nodes_interpolated[i]-1]
        time_levels[level_ind] += stay_time_original[i]

    first_half_time = np.sum(time_levels[:(len(time_levels)//2)])
    total_time = np.sum(time_levels)
    print("     % Time spent in first 1/2 levels: " + "{:.2f}".format(first_half_time/total_time*100)+ "%",end='\n\n') 
    
    # Stop time at each bin-----------------------------------------------------------------------------------------------------------------------------------------------------------
    stop_time_tmp = 1
    stop_time_record = []
    for i in range(1,len(behav_nodes_interpolated)):
        if behav_nodes_interpolated[i] == behav_nodes_interpolated[i-1]:
            stop_time_tmp+=1
        else:
            stop_time_record.append(stop_time_tmp)
            stop_time_tmp = 1
    # Delete first and last value: stop time at begining and end may have bias
    stop_time_record.pop(0)
    stop_time_record.pop(-1)

    plt.hist(stop_time_record, range = [0,300], bins = 30)
    stop_time_mean = np.mean(stop_time_record)
    stop_time_median = np.median(stop_time_record)

    path = os.path.join(totalpath,NumOfMice,date,"behav",'DwellTimeAtEachBin.pdf')
    plt.savefig(path)
    path = os.path.join(totalpath,NumOfMice,date,"behav",'DwellTimeAtEachBin.png')
    plt.savefig(path)
    print("     Figure 7 'DwellTimeAtEachBin' is done...")
    plt.close()
    
    # Save these results(Some key variables) with the form of pkl file. -------------------------------------------------------------------------------------------------------------------
    bin_travelled = 1
    for i in range(2,len(behav_nodes_interpolated)):
        if behav_nodes_interpolated[i] != behav_nodes_interpolated[i-1]:
            bin_travelled+=1
    speed_bin = bin_travelled / (total_time/1000)
    print("     ",speed_bin)
    path = os.path.join(totalpath,NumOfMice,date,"speed_bin.pkl")
    with open(path, 'wb') as f:
        pickle.dump([speed_bin], f)
    # change to cm/s 
    path = os.path.join(totalpath,NumOfMice,date,"behav_decision.pkl")
    with open(path, 'wb') as f:
        pickle.dump([correct_time/1000, wrong_time/1000, correct_time/(correct_time+wrong_time), decision_rate, time_levels], f)
        
    my_list = [behav_time_original, behav_nodes_interpolated, behav_dir]
    path = os.path.join(totalpath,NumOfMice,date,"behav_processed.pkl")
    with open(path, 'wb') as f:
        pickle.dump(my_list, f)    
    
    path = os.path.join(totalpath,NumOfMice,date,"stop_time.pkl")
    with open(path, 'wb') as f:
        pickle.dump([stop_time_mean, stop_time_median], f)
    
    print("     Every files have been saved! This session has done successfully!",end = '\n\n\n')
    # The END of MAIN FUNCTION |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def Generate_mean_value(mylist):
    file_path = mylist[4]
    # Read the video
    video_name = os.path.join(file_path,"behavCam1.avi")
    mean_frame = get_meanframe(video_name)
    return mean_frame

def Generate_Ori_position(mylist):
    file_path = mylist[4]    
    path = os.path.join(file_path,"behav_new.mat")
    behav_mat = loadmat(path); # read data
    row_correct = mylist[3]
    # 矫正开始、结束数据
    correct_time_data = pd.read_excel("G:\YSY\mice_maze_metadata_time_correction.xlsx", sheet_name = "training_recording_new")

    # position number 13
    behav_positions = behav_mat['behav']["position_original"][0][0]
    behav_time = behav_mat['behav']['time'][0][0]
    length = int(len(behav_positions))

    #手动设置删除帧数，存在“mice_maze_metadata_time_correction.xlsx”中
    delete_start = int(correct_time_data['start_time'][row_correct])
    delete_end = int(correct_time_data['end_time'][row_correct])

    behav_positions = np.delete(behav_positions, range(delete_end,length), axis = 0)
    behav_positions = np.delete(behav_positions, range(0,delete_start), axis = 0)
    behav_time = np.delete(behav_time, range(delete_end,length))
    behav_time = np.delete(behav_time, range(0,delete_start))
    # delete first 50 frames (~1.7s), when the mouse is not in the maze
    # should replace to exact starting time later!
    #     delete_start = 50
    #     behav_positions = np.delete(behav_positions, range(0,delete_start),0)
    #     behav_time = np.delete(behav_time, range(0,delete_start))

    # delete NAN values
    nan_mask = np.isnan(behav_positions).any(axis=1)
    behav_time_original = behav_time
    behav_positions = behav_positions[~nan_mask, ]
    behav_time = behav_time[~nan_mask]
    
    # data cleaning: delete wrong data point by filtering speed ------------------------------------------------------------------------
    behav_len = behav_positions.shape[0]
    nan_seq_len = 0
    delta = 0
    max_speed = 60 # 100 cm/s

    behav_positions_tmp = behav_positions
    behav_time_tmp = behav_time
    good_ind_diff = 1
    for i in range(1,behav_len):
        delta = np.sqrt(np.sum(np.square(behav_positions[i,] - behav_positions[i-good_ind_diff,])))
        time_diff = behav_time[i] - behav_time[i-good_ind_diff]
        speed_tmp = delta/time_diff*1000
        if speed_tmp > max_speed:
            # print(i, behav_positions[i,],behav_positions[i-1,], speed_tmp, time_diff, delta)
            behav_positions_tmp[i,] = np.nan
            good_ind_diff += 1
        else:
            good_ind_diff = 1

    nan_mask2 = np.isnan(behav_positions_tmp).any(axis=1)
    behav_positions = behav_positions_tmp[~nan_mask2, ]
    
    behav_roi = behav_mat['behav']['ROI'][0][0][0]
    track_length = behav_mat['behav']['trackLength'][0][0][0]
    ori_positions = behav_positions * behav_roi[2]/track_length +  [behav_roi[0], behav_roi[1]]
    return ori_positions

maxWidth = 360
maxHeight = 360
_coords_range = [[0,maxWidth +0.01],[0, maxHeight+0.01]]
_nbins = 12
FINAL_LINE_COLOR = (255, 100, 0)
WORKING_LINE_COLOR = (127, 127, 127)

import pandas as pd
file = pd.read_excel("G:\YSY\mice_maze_metadata_time_correction.xlsx", sheet_name = "training_recording_new")

for i in range(len(file)):
    print(file['number'][i], file['date'][i], file['maze_type'][i],"is calculating...........................................................")
    mylist = [str(file['date'][i]), str(file['number'][i]), int(file['maze_type'][i]), i, str(file['recording_folder'][i])]
    mean_frame = Generate_mean_value(mylist)
    equ_meanframe = cv2.equalizeHist(np.uint8(mean_frame))
    ori_positions = Generate_Ori_position(mylist)
    Run_all_mice(mylist) 

'''    date = mylist[0]
    NumOfMice = mylist[1]
    maze_type = mylist[2]
    row = mylist[3]
    row_correct = row - 2
    file_path = mylist[4]
'''    
        
    
    
    
    
    