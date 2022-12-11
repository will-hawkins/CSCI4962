import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import time
from datetime import datetime, timedelta
import networkx as nx
import random
from tqdm import tqdm

def to_int(x):
    try:
        return int(x)
    except:
        return np.nan
def to_dt(d):
    try:
        return datetime.strptime(d, "%Y-%m-%d %H:%M:%S.%f %Z")
    except:
        return datetime.strptime(d, "%Y-%m-%d %H:%M:%S %Z")

dt_func = np.vectorize(to_dt)

class Pixel:
    def __init__(self,ts, user, color, x_cor, y_cor):
        self.ts = ts
        self.user = user
        self.color = color
        self.x_cor = x_cor
        self.y_cor = y_cor

def canvas_at_time(canvas, T):
    X,Y = canvas.shape
    canv = np.empty([X,Y], dtype=object)
    for i in range(X):
        for j in range(Y):
            if len(canvas[i,j]) != 0:
                low = 0
                high = len(canvas[i,j])-1
                while (low != high):
                    mid = (low + high) // 2
                    if canvas[i,j][mid].ts <= T:
                        low = mid + 1
                    else:
                        high = mid
                canv[i,j] = canvas[i,j][high]
    return canv
def board_colors(canvas,colors):
    h,w = canvas.shape
    board = np.empty([h,w,3], dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            try:
                board[i,j] = colors[int(canvas[i,j])]
            except:
                board[i,j] = colors[0]
    return board

def board_colors_pixel(canvas,colors):
    h,w = canvas.shape
    board = np.empty([h,w,3], dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            try:
                board[i,j] = colors[canvas[i,j].pixel]
            except:
                board[i,j] = colors[0]
    return board

def time_lapse(canvas, start, stop, step):
    boards = []
    t = start
    while t <= stop:
        boards.append(board_colors(canvas_at_time(canvas,t ), colors))
        t += step
    return boards

if __name__ == '__main__':

	data_df = pd.read_csv("place_tiles.csv")
	ts = data_df['ts'].to_numpy()
	user = data_df['user_hash'].to_numpy()
	color = data_df['color'].to_numpy()
	x_cor = data_df['x_coordinate'].to_numpy()
	y_cor = data_df['y_coordinate'].to_numpy()

	x = time.time()
	ts = dt_func(ts)
	td = time.time() - x

	data = np.stack([ts,user,x_cor,y_cor,color],axis=1)

	canvas = np.empty( (1001,1001), dtype=object)
	for i in range(1001):
	    for j in range(1001):
	        canvas[j,i] = []
	x = time.time()
	for (ts, user, x_cor, y_cor, color) in data:
	    canvas[y_cor, x_cor].append( Pixel(ts,user,color,x_cor,y_cor) )
	td = time.time() - x
	print(td)


	canvas_counts = np.empty( (1001,1001), dtype=object)
	for i in range(1001):
	    for j in range(1001):
	        canvas_counts[i,j] = len(canvas[i,j])

	#sort pixels
	for i in range(1001):
	    for j in range(1001):
	        canvas[i,j].sort(key=lambda x: x.ts)


	users = {}
	for (_, user, _, _, _) in data:
	    users[user] = []

	t = time.time()
	for x in range(1001):
	    for y in range(1001):
	        for p in canvas[x,y]:
	            users[p.user].append(p)
	print(time.time()-t)

	### User Graph

	t = time.time()
	user_graph = nx.Graph()
	for x in range(1001):
	    for y in range(1001):
	        if len(canvas[x,y]) < 10:
	            for i in range(len(canvas[x,y])):
	                for j in range(i,len(canvas[x,y])):
	                    user_graph.add_edge(canvas[x,y][i].user, canvas[x,y][j].user)
