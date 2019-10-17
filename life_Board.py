import numpy as np
import scipy as sp
# from scipy import signal
import tensorflow as tf

class LifeBoard:
    def __init__(self, parent, x, y, life_probability):
        self.init_complete = False
        self.parent = parent
        self.neighbor_filter = np.array([[1,1,1],
                                         [1,0,1],
                                         [1,1,1]])
        self.history_depth = 50
        self.setup_board(x,y,0)
        self.init_complete = True
    def setup_board(self,x,y,life_probability,use_current=False): 
        if not use_current:
            self.x = x
            self.y = y
            if life_probability != 0: 
                inverse = 1 / life_probability
                sample = inverse * np.random.random((y,x))
            else: sample = np.ones((y,x))
            self.board = np.zeros((y,x))
            self.board[sample < 1] = 1
        self.history = None
        self.generation = 0
        self.oscillating = False
        self.oscillation_start = None
        self.oscillation_period = None
        self.board_static = False
        self.parent.display.boring_state_detected = False
        self.update_board_stats()
        if self.init_complete: self.parent.display.update_population_entry()
        self.init_tf_graph()
    def step(self,random_add_probability):
        if not self.oscillating and not self.board_static:
            # sums = sp.signal.convolve2d(self.board,self.neighbor_filter,mode='same')
            sums = self.sess.run([self.next_step],
                       feed_dict={self.current_step: self.board})
            sums = sums[0]
            sums = sums.reshape((self.y,self.x))
            
            live_cells = (self.board == 1)
            dead_cells = (self.board == 0)
            
            # Any live cell with fewer than two live neighbors dies, as if by underpopulation.
            less_than_two_neighbors = (sums < 2)
            self.board[np.logical_and(live_cells,less_than_two_neighbors)] = 0
            
            # Any live cell with two or three live neighbors lives on to the next generation.
            two_or_three_neighbors = np.logical_or(sums == 2,sums == 3)
            self.board[np.logical_and(live_cells,two_or_three_neighbors)] = 1
            
            # Any live cell with more than three live neighbors dies, as if by overpopulation.
            more_than_three_neighbors = (sums > 3)
            self.board[np.logical_and(live_cells,more_than_three_neighbors)] = 0
            
            # Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
            exactly_three_neighbors = (sums == 3)
            self.board[np.logical_and(dead_cells,exactly_three_neighbors)] = 1
            
            # Randomly add living cells to the board. Not for Life purists. Set to 0 in the GUI if you hate.
            random_life = np.random.random((self.y,self.x))
            where_add = (random_life < random_add_probability)
            self.board[where_add] = 1
        elif self.board_static: self.generation += 1
        else:
            history_location = self.oscillation_start + (self.generation - self.oscillation_start) % self.oscillation_period
            self.board = np.squeeze(self.history[history_location,...])
        
        self.update_board_stats()
    def update_board_stats(self):
        self.population = int(np.sum(self.board))
        self.generation += 1
        if not self.oscillating: self.update_history()
    def init_tf_graph(self):
        self.neighbor_filter = self.neighbor_filter.reshape((3,3,1,1))
        self.current_step = tf.placeholder(tf.float32, shape=(None,None))
        current_step_reshape = tf.reshape(self.current_step, shape=[1,self.y,self.x,1])
        self.next_step = tf.nn.conv2d(current_step_reshape, 
                                 self.neighbor_filter,
                                 strides = [1,1,1,1],
                                 padding='SAME')
        self.sess = tf.Session()
    def update_history(self):
        # check for oscillations
        if self.history is not None:
            if self.history.shape[0] > 1:
                matches = (self.board == self.history).astype(np.int32)
                matches = matches.reshape(len(self.history),self.board.size)
                sums = matches.sum(axis=1)
                where_all_match = np.argwhere(sums == self.board.size)
                if np.size(where_all_match) > 0:
                    self.oscillation_start = max(where_all_match)
                    self.oscillation_period = self.history.shape[0]-self.oscillation_start
                    if self.oscillation_period > 1: self.oscillating = True
                    else: self.board_static = True
        # add current board to history
        current_board = np.array(self.board)
        current_board = current_board.reshape([1]+list(self.board.shape))
        if self.history is None: self.history = current_board
        else:
            self.history = np.concatenate((self.history,current_board),axis=0)
            if len(self.history)>self.history_depth:
                amount_to_cut = len(self.history)-self.history_depth
                self.history = self.history[amount_to_cut:]