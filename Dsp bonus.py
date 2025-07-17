import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk



def load():
    fs=100
    global signal_samples1,signal_indices1
    signal_samples1=[]
    signal_indices1=[]
    A = simpledialog.askinteger("Input", "Enter the Amplitude :")
    if A is None:
        messagebox.showwarning("Invalid Input", "Please enter the Amplitude.")
        return
    f = simpledialog.askinteger("Input", "Enter the frequency:")
    if f is None:
        messagebox.showwarning("Invalid Input", "Please enter the frequency.")
        return

    for n in range(100):
        signal_indices1.append(n/100)
        signal_samples1.append(A*np.sin(2*np.pi*n*f/fs))
    return signal_samples1

def loadd(arr):
    global signal_samples1, signal_indices1
    signal_samples1 = []
    signal_samples1=arr





def correlation():

    N=len(signal_samples1)
    R1=[]
    res=[]



    for j in range(N):
        r=0




        for n in range(N):
            if(n-j>=0):
                r+=signal_samples1[n]*signal_samples1[(n-j)]



        R1.append(r)
    for x in range(100):
        res.append(R1[99-x])

    for x in range(100):
        res.append(R1[x])
    return res



def noise():
    n=np.random.normal(0,0.5,100)

    return n


def corrfinal():
    l= load()
    c=correlation()
    n=noise()
    noise_corr = n + l
    loadd(noise_corr)
    fin=correlation()




    messagebox.showinfo("Signal ", f"Done")
    r=range(-100,100)
    # Plot Frequency vs Phase
    plt.figure(figsize=(6, 4))

    plt.plot(signal_indices1, l, 'b')
    plt.title('sin')
    plt.xlabel('time')
    plt.ylabel('amp')
    plt.grid()

    # Plot Frequency vs Phase
    plt.figure(figsize=(6, 4))
    plt.plot(r, c, 'r')
    plt.title('corr')
    plt.xlabel('lag')
    plt.ylabel('corr')
    plt.grid()

    plt.figure(figsize=(6, 4))
    plt.plot(signal_indices1, n, 'r')
    plt.title('noise')
    plt.xlabel('time')
    plt.ylabel('amp')
    plt.grid()


    plt.figure(figsize=(6, 4))
    plt.plot(signal_indices1, noise_corr, 'r')
    plt.title('noise_corr')
    plt.xlabel('time')
    plt.ylabel('amp')
    plt.grid()


    plt.figure(figsize=(6, 4))
    plt.plot(r, fin, 'r')
    plt.title('corr for noise_corr')
    plt.xlabel('lag')
    plt.ylabel('corr')
    plt.grid()


    plt.show()


root = tk.Tk()
root.title("Signal Processing GUI")
root.geometry("500x400")
root.config(bg="#f0f0f0")
notebook = ttk.Notebook(root)
notebook.pack(pady=10, fill='both', expand=True)

# Create frames for the tabs
task1_frame = tk.Frame(notebook, bg="#f0f0f0")
task2_frame = tk.Frame(notebook, bg="#f0f0f0")
task3_frame = tk.Frame(notebook, bg="#f0f0f0")
task5_frame = tk.Frame(notebook, bg="#f0f0f0")
task6_tab = ttk.Frame(notebook)
task7_tab = ttk.Frame(notebook)

# Add frames to notebook
notebook.add(task1_frame, text='bonus Task')

# Layout frames for Task 1
top_frame = tk.Frame(task1_frame, bg="#f0f0f0")
top_frame.pack(pady=10)

middle_frame = tk.Frame(task1_frame, bg="#f0f0f0")
middle_frame.pack(pady=10)

bottom_frame = tk.Frame(task1_frame, bg="#f0f0f0")
bottom_frame.pack(pady=10)





# Function to create and place buttons
def create_button(frame, text, command, row, column, width=20, bg="#4CAF50", fg="white"):
    button = tk.Button(frame, text=text, command=command, width=width, bg=bg, fg=fg)
    button.grid(row=row, column=column, padx=10, pady=5)
    return button


# Action buttons for Task 1
create_button(middle_frame, "correlation", corrfinal, 0, 0, bg="#4CAF50")
create_button(bottom_frame, "Exit", root.quit, 1, 0, bg="#F44336")



root.mainloop()
