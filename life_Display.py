from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import Canvas,Tk,ttk,Label,Entry,Button,mainloop,Text,Frame,IntVar,Checkbutton
import os
import numpy as np
from tkinter import filedialog
import math
import random
import time

class Display:
    def __init__(self, parent):
        self.parent = parent
        self.main_font = ("Courier", 22, "bold")
        self.max_win_size = (1335,950)
        self.canvas_size = ((self.max_win_size[0]-5,self.max_win_size[1]-230))
        self.canvas_image_counter = 0
        self.im = {}
        self.setup_window()
        self.boring_state_detected_at = 0
        self.make_canvas_interactive()
    def open_images(self):
        pil_img = Image.open('source/play.gif').resize((80,80), Image.ANTIALIAS)
        self.play_photo=ImageTk.PhotoImage(pil_img)
        pil_img = Image.open('source/pause.gif').resize((80,80), Image.ANTIALIAS)
        self.pause_photo=ImageTk.PhotoImage(pil_img)
        pil_img = Image.open('source/close.gif').resize((80,80), Image.ANTIALIAS)
        self.close_photo=ImageTk.PhotoImage(pil_img)
        pil_img = Image.open('source/accept.gif').resize((80,80), Image.ANTIALIAS)
        self.accept_photo=ImageTk.PhotoImage(pil_img)
        pil_img = Image.open('source/refresh.gif').resize((80,80), Image.ANTIALIAS)
        self.refresh_photo=ImageTk.PhotoImage(pil_img)
        pil_img = Image.open('source/random.gif').resize((80,80), Image.ANTIALIAS)
        self.random_photo=ImageTk.PhotoImage(pil_img)
        pil_img = Image.open('source/next.gif').resize((80,80), Image.ANTIALIAS)
        self.next_photo=ImageTk.PhotoImage(pil_img)
        pil_img = Image.open('source/trash.gif').resize((80,80), Image.ANTIALIAS)
        self.clear_photo=ImageTk.PhotoImage(pil_img)
    def setup_window(self):
        # initial setup
        self.primary_window = Tk()
        self.open_images()
        self.primary_window.wm_title("Life")
        self.primary_window.geometry('1274x960-1+0')
        self.primary_window.minsize(width=100, height=30)
        self.primary_window.maxsize(width=self.max_win_size[0], height=self.max_win_size[1])
        
        # image & canvas
        
        self.im_frame = ttk.Frame(self.primary_window)
        self.im_frame.grid(row=0,column=0,columnspan=2,sticky="nsew")
        self.im_frame.columnconfigure(0, weight=1)
        self.im_frame.rowconfigure(0, weight=1)
        self.primary_window.columnconfigure(0, weight=1)
        self.primary_window.rowconfigure(0, weight=1)
        
        self.canvas_frame = ttk.Frame(self.primary_window)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew")
        self.canvas_frame.columnconfigure(0, weight=1)
        
        self.the_canvas = Canvas(self.canvas_frame,
                                width=self.canvas_size[0],
                                height=self.canvas_size[1],
                                background='black')
        self.the_canvas.grid(row=0, column=0,columnspan=2, sticky="ew")
        
        # bottom buttons
        self.bottom_buttons_frame = ttk.Frame(self.primary_window)
        self.bottom_buttons_frame.grid(row=3,column=0,columnspan=2)
        #
        self.play_button = Button(self.bottom_buttons_frame,
                                    command= self.play_button_func,
                                    image=self.play_photo,
                                    width="80",height="80")
        self.play_button.grid(row=0,column=0)
        #
        self.pause_button = Button(self.bottom_buttons_frame,
                                      command=self.pause_button_func,
                                      image=self.pause_photo,
                                      width="80",height="80")
        self.pause_button.grid(row=0,column=2)
        #
        self.random_button = Button(self.bottom_buttons_frame,
                                    command=self.random_button_func,
                                    image=self.random_photo,
                                    width="80",height="80")
        self.random_button.grid(row=0,column=3)
        #
        self.next_button = Button(self.bottom_buttons_frame,
                                   command=self.next,
                                   image=self.next_photo,
                                   width="80",height="80")
        self.next_button.grid(row=0,column=4)
        #
        self.clear_button = Button(self.bottom_buttons_frame,
                                   command=self.clear,
                                   image=self.clear_photo,
                                   width="80",height="80")
        self.clear_button.grid(row=0,column=5)
        #
        self.close_button = Button(self.bottom_buttons_frame,
                                   command=self.parent.close,
                                   image=self.close_photo,
                                   width="80",height="80")
        self.close_button.grid(row=0,column=6)
        # bottom entries 0
        self.bottom_entries_frame0 = ttk.Frame(self.primary_window)
        self.bottom_entries_frame0.grid(row=4,column=0,columnspan=2)
        #
        Label(self.bottom_entries_frame0, text="Generation Number: ",font=self.main_font).grid(row=0, column=0)
        self.generation_entry = Entry(self.bottom_entries_frame0,justify='right')
        self.generation_entry.insert("end","1")
        self.generation_entry.config(state="disabled",font=self.main_font,width=5)
        self.generation_entry.grid(row=0,column=1)
        #
        Label(self.bottom_entries_frame0, text=" Population: ",font=self.main_font).grid(row=0, column=3)
        self.population = Entry(self.bottom_entries_frame0,justify='right')
        self.population.insert("end","0")
        self.population.config(state="disabled",font=self.main_font,width=5)
        self.population.grid(row=0,column=4)
        #
        Label(self.bottom_entries_frame0, text=" State: ",font=self.main_font).grid(row=0, column=5)
        self.state = Entry(self.bottom_entries_frame0,justify='center')
        self.state.insert("end","Normal")
        self.state.config(state="disabled",font=self.main_font,width=17)
        self.state.grid(row=0,column=6)
        # bottom entries 1
        self.bottom_entries_frame1 = ttk.Frame(self.primary_window)
        self.bottom_entries_frame1.grid(row=5,column=0,columnspan=2)
        #
        #
        self.grid_check = IntVar()
        self.grid_check.set(1)
        self.grid_button = Checkbutton(self.bottom_entries_frame1, text="Grid", variable=self.grid_check,font=self.main_font)
        self.grid_button.grid(row=0,column=1)
        #
        Label(self.bottom_entries_frame1, text=" Width:",font=self.main_font).grid(row=0, column=4)
        self.width_entry = Entry(self.bottom_entries_frame1,justify='right')
        # self.width_entry.insert("end", str(self.canvas_size[0]//3))
        self.width_entry.insert("end", '100')
        self.width_entry.config(font=self.main_font,width=5)
        self.width_entry.grid(row=0,column=5)
        #
        Label(self.bottom_entries_frame1, text=" Height:",font=self.main_font).grid(row=0, column=6)
        self.height_entry = Entry(self.bottom_entries_frame1,justify='right')
        # self.height_entry.insert("end", str(self.canvas_size[1]//3))
        self.height_entry.insert("end", '60')
        self.height_entry.config(font=self.main_font,width=5)
        self.height_entry.grid(row=0,column=7)
        #
        Label(self.bottom_entries_frame1, text=" Life Probability:",font=self.main_font).grid(row=0, column=8)
        self.life_probability = Entry(self.bottom_entries_frame1,justify='right')
        self.life_probability.insert("end", '.42')
        self.life_probability.config(font=self.main_font,width=3)
        self.life_probability.grid(row=0,column=9)
        # bottom entries 2
        self.bottom_entries_frame2 = ttk.Frame(self.primary_window)
        self.bottom_entries_frame2.grid(row=6,column=0,columnspan=2)
        #
        self.Image_ANTIALIAS_check = IntVar()
        self.Image_ANTIALIAS_check.set(0)
        self.Image_ANTIALIAS_button = Checkbutton(self.bottom_entries_frame2, text="Image.ANTIALIAS ", variable=self.Image_ANTIALIAS_check,font=self.main_font)
        self.Image_ANTIALIAS_button.grid(row=0,column=1)
        #
        ttk.Separator(self.bottom_entries_frame2, orient="vertical").grid(row=0, column=2, rowspan=1, sticky="ns")
        #
        Label(self.bottom_entries_frame2, text=" Wait Per Generation:",font=self.main_font).grid(row=0, column=3)
        self.time_entry = Entry(self.bottom_entries_frame2,justify='right')
        self.time_entry.insert("end", '0')
        self.time_entry.config(font=self.main_font,width=5)
        self.time_entry.grid(row=0,column=4)
        #
        Label(self.bottom_entries_frame2, text=" Add Probability:",font=self.main_font).grid(row=0, column=5)
        self.random_add_probability = Entry(self.bottom_entries_frame2,justify='right')
        self.random_add_probability.insert("end", '0')
        self.random_add_probability.config(font=self.main_font,width=7)
        self.random_add_probability.grid(row=0,column=6)
    def update_generation_entry(self):
        val = str(self.parent.board.generation)
        self.generation_entry.config(state="normal")
        self.generation_entry.delete(0,'end')
        self.generation_entry.insert('end',val)
        self.generation_entry.config(state='disabled')
    def update_population_entry(self):
        val = str(self.parent.board.population)
        self.population.config(state="normal")
        self.population.delete(0,'end')
        self.population.insert('end',val)
        self.population.config(state='disabled')
    def update_state_entry(self):
        if self.parent.board.oscillating:
            period = str(self.parent.board.oscillation_period)
            self.board_state = 'Oscillating-' + period
        elif self.parent.board.board_static: self.board_state = 'Static'
        else: self.board_state = 'Normal'
        if self.board_state == 'Oscillating-[2]' or self.board_state == 'Oscillating-[6]':
            if not self.boring_state_detected: 
                self.boring_state_detected_at = time.time()
                self.boring_state_detected = True
            if (time.time() - self.boring_state_detected_at) > 10:
                self.random_button_func()
        if self.board_state == self.state.get(): return
        self.state.config(state="normal")
        self.state.delete(0,'end')
        self.state.insert('end',self.board_state)
        self.state.config(state='disabled')
    def play_button_func(self):
        if self.parent.pause: self.parent.pause = False
        else: self.parent.main_queue.put(self.play)
    def play(self):
        while True:
            while self.parent.pause == True: 
                print("pausing...",end="\r")
                time.sleep(.25)
            if self.parent.board.oscillating:
                self.parent.board.step(None)
            else:
                try: 
                    probability = float(self.random_add_probability.get())
                    self.old_probability = probability
                except: probability = self.old_probability
                self.parent.board.step(probability)
            #
            self.update_generation_entry()
            self.update_population_entry()
            self.update_state_entry()
            #
            self.update_display2()
            
            try: 
                time_entry = self.time_entry.get()
                time_entry = float(self.time_entry.get())
                self.old_time_entry = time_entry
            except:
                time_entry = self.old_time_entry
            time.sleep(time_entry)
    def pause_button_func(self):
        if self.parent.pause: self.parent.pause = False
        else: self.parent.pause = True
    def random_button_func(self):
        y = int(self.height_entry.get())
        x = int(self.width_entry.get())
        prob = float(self.life_probability.get())
        self.parent.board.setup_board(x,y,prob)
        self.board_state = 'Normal'
        #
        self.update_generation_entry()
        self.update_population_entry()
        self.update_state_entry()
        #
        if self.grid_check.get(): self.create_grid()
        self.make_canvas_interactive()
        self.update_display2()
    def next(self):
        self.parent.board.step(float(self.random_add_probability.get()))
        #
        self.update_generation_entry()
        self.update_population_entry()
        self.update_state_entry()
        #
        self.update_display2()
    def clear(self):
        y = int(self.height_entry.get())
        x = int(self.width_entry.get())
        self.parent.board.setup_board(x,y,0)
        self.board_state = 'Normal'
        #
        self.update_generation_entry()
        self.update_population_entry()
        self.update_state_entry()
        #
        if self.grid_check.get(): self.create_grid()
        self.make_canvas_interactive()
        self.update_display2()
    def create_grid(self):
        rows = int(self.height_entry.get())
        row_size = self.canvas_size[1]/rows
        columns = int(self.width_entry.get())
        column_size = self.canvas_size[0]/columns
        self.grid_im = Image.new('RGBA',self.canvas_size,(0,0,0,0))
        draw = ImageDraw.Draw(self.grid_im)
        for row in range(rows-1):
            i=0,(1+row)*row_size
            f=self.canvas_size[0],(1+row)*row_size
            draw.line((i[0],i[1],
                       f[0],f[1]),
                       fill="#303030")
        for column in range(columns-1):
            i=(1+column)*column_size,0
            f=(1+column)*column_size,self.canvas_size[1]
            draw.line((i[0],i[1],
                       f[0],f[1]),
                     fill="#303030")
    def update_display2(self):
        rows = int(self.height_entry.get())
        columns = int(self.width_entry.get())
        im = np.zeros((rows,columns,3),np.uint8)
        # if history_number is not None: board_to_display = self.parent.board.history[history_number,...]
        # else: board_to_display = self.parent.board.board
        cell_rows, cell_columns = np.where(self.parent.board.board == 1)
        number_of_cells=len(cell_rows)
        for i in range(number_of_cells):
            im[cell_rows[i],cell_columns[i]]=(0,0,255)
        im = Image.fromarray(im)
        if self.Image_ANTIALIAS_check.get() == 1: im = im.resize(self.canvas_size,Image.ANTIALIAS)
        else: im = im.resize(self.canvas_size)
        
        if self.grid_check.get(): im.paste(self.grid_im, (0,0), mask=self.grid_im)
        # im.save('im_test.gif')
        self.im[self.canvas_image_counter] = ImageTk.PhotoImage(im)
        self.the_canvas.create_image((0,0),anchor='nw',image=self.im[self.canvas_image_counter])
        
        if self.canvas_image_counter == 0: self.canvas_image_counter = 1
        else: self.canvas_image_counter = 0
    def make_canvas_interactive(self):
        self.last_cell_altered = None
        rows = int(self.height_entry.get())
        columns = int(self.width_entry.get())
        row_size = self.canvas_size[1]/rows
        column_size = self.canvas_size[0]/columns
        def callback(event,dragging):
            self.same_event = False
            board_location = (int(event.y / row_size),int(event.x / column_size))
            if not dragging:
                if self.parent.board.board[board_location] == 0: self.interaction_mode = 'add'
                else: self.interaction_mode = 'remove'
            if self.interaction_mode == 'add':
                if self.parent.board.board[board_location] == 0:
                    self.parent.board.board[board_location]=1
                    self.update_display2()
                    self.last_cell_altered = board_location
            if self.interaction_mode == 'remove':
                if self.parent.board.board[board_location] == 1:
                    self.parent.board.board[board_location]=0
                    self.update_display2()
                    self.last_cell_altered = board_location
        def refresh_after_button_press(event):
            self.parent.board.setup_board(None,None,None,True)
            self.board_state = 'Normal'
        self.the_canvas.bind("<Button-1>", lambda event, dragging=False: callback(event,dragging))
        self.the_canvas.bind("<B1-Motion>", lambda event, dragging=True: callback(event,dragging))
        self.the_canvas.bind("<ButtonRelease-1>",refresh_after_button_press)