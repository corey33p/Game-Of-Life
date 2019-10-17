from life_Display import Display
from life_Board import LifeBoard
from tkinter import mainloop
import os
import win32gui
import win32con
import threading
from queue import Queue
import time
import time

class Parent:
    def __init__(self):
        self.main_queue = Queue()
        self.display  = Display(self)
        self.board = LifeBoard(self,
                               int(self.display.width_entry.get()),
                               int(self.display.height_entry.get()),
                               float(self.display.life_probability.get()))
        self.display.update_population_entry()
        if self.display.grid_check.get(): self.display.create_grid()
        self.display.update_display2()
        self.resize_CLI_window()
        self.pause = False
        main_queue_thread = threading.Thread(target=lambda: self. main_queue_thread())
        main_queue_thread.daemon = True
        main_queue_thread.start()
        mainloop()
    def resize_CLI_window(self):
        def get_windows():
            def check(hwnd, param):
                title = win32gui.GetWindowText(hwnd)
                if 'life_Main' in title and 'Notepad++' not in title:
                    param.append(hwnd)
            wind = []
            win32gui.EnumWindows(check, wind)
            return wind
        self.cli_handles = get_windows()
        for window in self.cli_handles:
            win32gui.MoveWindow(window,0,0,self.display.max_win_size[0],self.display.max_win_size[1],True)
    def main_queue_thread(self):
        while True: # handle objects in the queue until game_lost
            time.sleep(.25)
            try:
                next_action = self.main_queue.get(False)
                next_action()
            except Exception as e: 
                if str(e) != "": print(e)
        self.main_queue.queue.clear()
        self.close()
    def close(self):
        for handle in self.cli_handles:
            win32gui.PostMessage(handle,win32con.WM_CLOSE,0,0)

if __name__ == '__main__':
    main_object = Parent()
    