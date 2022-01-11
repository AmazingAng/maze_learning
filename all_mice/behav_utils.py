import matplotlib.pyplot as plt
import pickle
import cv2
import numpy as np
import time
import scipy.stats

FINAL_LINE_COLOR = (255, 100, 0)
WORKING_LINE_COLOR = (127, 127, 127)

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


class PolygonDrawer(object):
    def __init__(self, window_name, equ_meanframe, ori_positions):
        self.window_name = window_name # Name for our window

        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.equ_meanframe = equ_meanframe
        self.ori_positions = ori_positions
    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every smouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True


    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, self.equ_meanframe)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            canvas = self.equ_meanframe
            # canvas = np.zeros(CANVAS_SIZE, np.uint8)
            cv2.polylines(canvas,  np.int32([self.ori_positions]), False, FINAL_LINE_COLOR, 1)

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
        canvas = self.equ_meanframe

        # of a filled polygon
        if (len(self.points) > 0):
            cv2.polylines(canvas, np.array([self.points]),True, FINAL_LINE_COLOR, thickness = 5)
        # And show it
        cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()
        
        # Four points transform
        warped_image, M = four_point_transform(self.equ_meanframe, np.asarray(self.points))
        cv2.imshow("Processed Maze", warped_image)
        warped_positions = cv2.perspectiveTransform(np.array([self.ori_positions]) , M)[0]
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        cv2.destroyWindow("Processed Maze")
       
        return warped_image, warped_positions, M

    
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


def get_direction(behav_nodes_interpolated, test_maze, shortest_path, cell_dists):
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


def transform_bin(bin_numbers):
    # rotate bin by 90 degree
    y = 13 - bin_numbers[1,:]
    x = bin_numbers[0,:]
    return np.array([x,y])

def node_to_run(node, runs):
    # transform node number (start from 1) to run number (start from 1)
    # for example node 1, to run 1
    run_ind = 0
    for run_i in runs:
        if node in run_i:
            run_ind = runs.index(run_i) +1
            return run_ind
    return run_ind
        