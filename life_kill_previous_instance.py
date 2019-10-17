import win32gui
import win32con
def get_windows():
    def check(hwnd, param):
        title = win32gui.GetWindowText(hwnd)
        if 'life_Main' in title and 'Notepad++' not in title:
            param.append(hwnd)
    wind = []
    win32gui.EnumWindows(check, wind)
    return wind
cli_handles = get_windows()
for handle in cli_handles:
    win32gui.PostMessage(handle,win32con.WM_CLOSE,0,0)
