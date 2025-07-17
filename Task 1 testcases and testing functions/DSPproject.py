import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from TestCase import *

global islevel
class ChoiceDialog:
    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.top.title("Choose an Option")

        tk.Label(self.top, text="Please choose continuous or discrete :").pack(pady=10)

        self.option1_button = tk.Button(self.top, text="continuous", command=lambda: self.choose("continuous"))
        self.option1_button.pack(pady=5)

        self.option2_button = tk.Button(self.top, text="discrete", command=lambda: self.choose("discrete"))
        self.option2_button.pack(pady=5)

        self.result = None

    def choose(self, option):
        self.result = option
        self.top.destroy()  # Close the dialog

def open_choice_dialog():
    dialog = ChoiceDialog(root)
    root.wait_window(dialog.top)  # Wait for the dialog to close

    return dialog.result


def ReadSignalFile(file_name):
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=float(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    return expected_indices,expected_samples
def ReadFile(file_name):

    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()

        while line:
            # process line
            L=line.strip()
            V2=float(L)


            expected_samples.append(V2)
            line = f.readline()
    return expected_samples

def plot_signal(indices, values, title='Signal'):
    plt.figure(figsize=(10, 5))
    plt.stem(indices, values)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Value')
    plt.axhline(0, color='black')
    plt.grid(True)
    plt.show()


# Signal processing functions (includes all previous ones)
def add_signals(*signals, output_file='addOutput.txt'):
    all_indices = np.unique(np.concatenate([indices for indices, _ in signals]))
    result_values = np.zeros(len(all_indices))
    for indices, values in signals:
        interpolated_values = np.interp(all_indices, indices, values, left=0, right=0)
        result_values += interpolated_values
    result_values = result_values.astype(int)

    with open(output_file, 'w') as f:
        f.write('Index,Value\n')
        for idx, val in zip(all_indices, result_values):
            f.write(f"{idx},{val}\n")

    return all_indices, result_values


def amplify_signal(indices, values, factor, output_file='amplified_out.txt'):
    values = np.array(values)
    amplified_values = values * factor
    amplified_values = amplified_values.astype(int)

    with open(output_file, 'w') as f:
        f.write('Index,Amplified Value\n')
        for idx, val in zip(indices, amplified_values):
            f.write(f"{idx},{val}\n")

    return indices, amplified_values


def subtract_signals(signal1, signal2, output_file='sub_output.txt'):
    indices1, values1 = signal1
    indices2, values2 = signal2
    interpolated_values2 = np.interp(indices1, indices2, values2, left=0, right=0)
    result_values = values1 - interpolated_values2

    with open(output_file, 'w') as f:
        f.write('Index,Value\n')
        for idx, val in zip(indices1, result_values):
            f.write(f"{idx},{val}\n")

    return indices1, result_values


def shift_signal(indices, values, shift_amount, output_file='shift_output.txt'):
    indices = np.array(indices)
    shifted_indices = indices - shift_amount

    with open(output_file, 'w') as f:
        f.write('Shifted Index,Value\n')
        for idx, val in zip(shifted_indices, values):
            f.write(f"{idx},{val}\n")

    return shifted_indices, values


def fold_signal(indices, values, output_file='fold_output.txt'):
    negated_indices = -np.array(indices)
    sorted_indices = np.sort(negated_indices)
    sorted_order = np.argsort(negated_indices)
    sorted_values = np.array(values)[sorted_order]

    with open(output_file, 'w') as f:
        f.write('Folded Index,Value\n')
        for idx, val in zip(sorted_indices, sorted_values):
            f.write(f"{idx},{val}\n")

    return sorted_indices.tolist(), sorted_values.tolist()


# Signal Generation Functions
def generate_signal(wave_type, amplitude, phase_shift, frequency, sampling_frequency):
    t = np.arange(0, 1, 1 / sampling_frequency)  # Generate time vector for 1 second
    if wave_type == "sine":
        values = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)
    elif wave_type == "cosine":
        values = amplitude * np.cos(2 * np.pi * frequency * t + phase_shift)
    return t, values


# GUI functions
signal_indices = []
signal_samples = []


def load_signal():
    global signal_indices, signal_samples
    file_path = filedialog.askopenfilename(title="Select a Signal File", filetypes=[("Text Files", "*.txt")])
    if file_path:
        try:
            signal_indices, signal_samples = ReadSignalFile(file_path)
            if len(signal_indices) > 0:
                signal_label.config(text=f"Signal Loaded: {file_path.split('/')[-1]} (Length: {len(signal_samples)})")
                messagebox.showinfo("Signal Loaded", f"Signal loaded from {file_path}")
            else:
                raise ValueError("Empty signal data.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load signal: {e}")
            signal_label.config(text="No signal loaded.")


def plot_signal_gui():
    if signal_indices and signal_samples:
        plot_signal(signal_indices, signal_samples, title='Loaded Signal')
    else:
        messagebox.showwarning("No Signal", "Please load a signal first.")


def amplify_signal_gui():
    if signal_indices and signal_samples:
        factor = simpledialog.askfloat("Input", "Enter the amplification factor:")
        if factor is not None:
            amplified_indices, amplified_values = amplify_signal(signal_indices, signal_samples, factor)
            plot_signal(amplified_indices, amplified_values, title=f'Amplified Signal by {factor}')
    else:
        messagebox.showwarning("No Signal", "Please load a signal first.")


def add_signal_gui():
    file_path2 = filedialog.askopenfilename(title="Select the Second Signal File", filetypes=[("Text Files", "*.txt")])
    if file_path2:
        try:
            indices2, samples2 = ReadSignalFile(file_path2)
            added_indices, added_values = add_signals((signal_indices, signal_samples), (indices2, samples2))
            plot_signal(added_indices, added_values, title='Added Signals')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add signals: {e}")


def subtract_signal_gui():
    file_path2 = filedialog.askopenfilename(title="Select the Second Signal File", filetypes=[("Text Files", "*.txt")])
    if file_path2:
        try:
            indices2, samples2 = ReadSignalFile(file_path2)
            subtracted_indices, subtracted_values = subtract_signals((signal_indices, signal_samples),
                                                                     (indices2, samples2))
            plot_signal(subtracted_indices, subtracted_values, title='Subtracted Signals')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to subtract signals: {e}")


def shift_signal_gui():
    if signal_indices and signal_samples:
        shift_amount = simpledialog.askinteger("Input", "Enter the shift value:")
        if shift_amount is not None:
            shifted_indices, shifted_values = shift_signal(signal_indices, signal_samples, shift_amount)
            plot_signal(shifted_indices, shifted_values, title=f'Shifted Signal by {shift_amount}')
    else:
        messagebox.showwarning("No Signal", "Please load a signal first.")


def fold_signal_gui():
    if signal_indices and signal_samples:
        folded_indices, folded_samples = fold_signal(signal_indices, signal_samples)
        plot_signal(folded_indices, folded_samples, title='Folded Signal x(-n)')
    else:
        messagebox.showwarning("No Signal", "Please load a signal first.")


def clear_signal():
    global signal_indices, signal_samples
    signal_indices.clear()
    signal_samples.clear()
    signal_label.config(text="No signal loaded.")
    messagebox.showinfo("Signal Cleared", "The signal has been cleared.")
    plt.close('all')


# GUI Function to Collect Input and Plot
def generate_signal_gui(wave_type):
    # Get user input
    amplitude = simpledialog.askfloat("Input", "Enter the amplitude A:")
    phase_shift = simpledialog.askfloat("Input", "Enter the phase shift Î¸ (in radians):")
    frequency = simpledialog.askfloat("Input", "Enter the analog frequency (Hz):")
    sampling_frequency = simpledialog.askfloat("Input", "Enter the sampling frequency (Hz):")

    # Check Nyquist condition
    if sampling_frequency <= 2 * frequency:
        messagebox.showerror("Error", "Sampling frequency must be greater than twice the analog frequency.")
        return

    # Generate the discrete signal
    t_sampled, y_sampled = generate_signal(wave_type, amplitude, phase_shift, frequency, sampling_frequency)

    # Generate the continuous signal for plotting purposes
    t_continuous = np.linspace(0, 1, 1000)  # High-resolution time axis
    if wave_type == "sine":
        y_continuous = amplitude * np.sin(2 * np.pi * frequency * t_continuous + phase_shift)
    elif wave_type == "cosine":
        y_continuous = amplitude * np.cos(2 * np.pi * frequency * t_continuous + phase_shift)

    # Plotting the signals
    plt.figure(figsize=(10, 5))

    # Plot the continuous wave
    plt.plot(t_continuous, y_continuous, label=f'Continuous {wave_type.capitalize()} Wave', color='blue')

    # Plot the discrete sampled points (without use_line_collection)
    plt.stem(t_sampled, y_sampled, linefmt='r-', markerfmt='ro', basefmt='k-', label='Discrete Samples')

    # Customize the plot
    plt.title(f'{wave_type.capitalize()} Wave with Sampled Points')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.show()

def ask_choice():
    response = messagebox.askyesno("Choose", "Do you want to display another one?")
    if response:
        return True
    else:
       return False
def displaySignal():
    global signal_indices1, signal_samples1,signal_indices2, signal_samples2
    check1=True
    file_path = filedialog.askopenfilename(title="Select a Signal File", filetypes=[("Text Files", "*.txt")])
    if file_path:
        try:
            signal_indices1, signal_samples1 = ReadSignalFile(file_path)
            if len(signal_indices1) > 0:
                check2=ask_choice()

                if(check2==True):
                    file_path2 = filedialog.askopenfilename(title="Select a Signal File",
                                                           filetypes=[("Text Files", "*.txt")])
                    if file_path2:
                        try:
                            signal_indices2, signal_samples2 = ReadSignalFile(file_path2)
                            if len(signal_indices2) > 0:
                                messagebox.showinfo("Signal ", f"Done")

                            else:
                                 raise ValueError("Empty signal data.")
                        except Exception as e:
                            messagebox.showerror("Error", f"Failed to display signal: {e}")
                            signal_label.config(text="No signal .")
                            check1 = False
                    else:
                        check2=False
            else:
                raise ValueError("Empty signal data.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display signal: {e}")
            signal_label.config(text="No signal .")
            check1=False
    else:
        return
    if(check1):
        opt = open_choice_dialog()


        plt.figure(figsize=(10, 5))
        if (opt == "continuous"):
            plt.plot(signal_indices1, signal_samples1)
        else:
            plt.stem(signal_indices1, signal_samples1)
        plt.title("signal")
        plt.xlabel('Sample Index')
        plt.ylabel('Sample Value')
        plt.grid(True)

        if (check2):
            plt.figure(figsize=(10, 5))
            if (opt == "continuous"):
                plt.plot(signal_indices2, signal_samples2)
            else:
                plt.stem(signal_indices2, signal_samples2)
            plt.title("signal")
            plt.xlabel('Sample Index')
            plt.ylabel('Sample Value')
            plt.grid(True)




        plt.show()


import numpy as np



def decimal_to_binary_with_bits(decimal_num, num_bits):
    binary_num = bin(decimal_num)[2:]  # Convert decimal to binary and remove '0b' prefix
    padding = '0' * max(0, num_bits - len(binary_num))  # Calculate padding
    return padding + binary_num

def quantize_signal(indices, values, levels,islevel):

    values = np.array(values)
    min_val, max_val = np.min(values), np.max(values)

    # Calculate step size with level adjustment
    step_size = (max_val - min_val) / levels
    interval=np.zeros(levels+1)
    quantized_values=np.zeros(len(values))
    err = np.zeros(len(values))
    inter=np.zeros(len(values))
    encoded_values=[]

    for i in range(levels+1):
        interval[i] = np.round(min_val + step_size * i,decimals=3)
    c=-1
    #for j in range(levels+1):
        #print(interval[j])
    for i in values:
        c+=1
        for j in range(levels):
            if i >=interval[j] and i<=interval[j+1]:
                encoded_values.append(decimal_to_binary_with_bits(j, int(np.log2(levels))))
                quantized_values[c]=(interval[j]+interval[j + 1])/2
                err[c]=quantized_values[c]-i
                inter[c]=j+1
                break



    # Perform quantization
    #print(inter)
    # Interpolate to create smooth representations of original and quantized signals
    continuous_indices = np.linspace(np.min(indices), np.max(indices), 1000)  # Fine resolution
    continuous_original = np.interp(continuous_indices, indices, values)
    continuous_quantized = np.interp(continuous_indices, indices, quantized_values)

    # Encode the quantized values

    #print("Encoded Values:", encoded_values,"quantized_values:", quantized_values,"error_values:", err)
    # Call the QuantizationTest1 function

    if(islevel==True):
        QuantizationTest2("Quan2_Out.txt",inter,encoded_values,quantized_values,err)
    else:
        QuantizationTest1("Quan1_Out.txt", encoded_values, quantized_values)  # File handling inside this function
    return continuous_indices, continuous_original, continuous_quantized


class QuantizeChoiceDialog:
    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.top.title("Quantization Choice")

        self.bits_var = tk.BooleanVar()
        self.levels_var = tk.BooleanVar()

        tk.Checkbutton(self.top, text="Number of Bits", variable=self.bits_var).pack(pady=5)
        tk.Checkbutton(self.top, text="Number of Levels", variable=self.levels_var).pack(pady=5)

        tk.Button(self.top, text="Submit", command=self.submit).pack(pady=10)

        self.result = None

    def submit(self):
        if self.bits_var.get() and not self.levels_var.get():
            self.result = "bits"
        elif self.levels_var.get() and not self.bits_var.get():
            self.result = "levels"
        else:
            messagebox.showwarning("Warning", "Please select only one option.")
            return
        self.top.destroy()  # Close the dialog


def plot_original_and_quantized(continuous_indices, continuous_original, continuous_quantized):
    plt.figure(figsize=(10, 5))

    # Plot Continuous Original Signal
    plt.plot(continuous_indices, continuous_original, label='Original Signal', color='blue')

    # Plot Continuous Quantized Signal
    plt.plot(continuous_indices, continuous_quantized, label='Quantized Signal', color='red')

    plt.title('Original vs Quantized Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Value')
    plt.grid(True)
    plt.legend()
    plt.show()


def quantize_signal_gui():

    islevel=True
    load_signal()  # Use the existing load_signal function
    if signal_indices and signal_samples:
        dialog = QuantizeChoiceDialog(root)
        root.wait_window(dialog.top)  # Wait for the dialog to close

        if dialog.result == "bits":
            bits = simpledialog.askinteger("Input", "Enter number of bits:")
            if bits is not None:
                levels = 2 ** bits
                islevel = False
            else:

                return
        elif dialog.result == "levels":
            levels = simpledialog.askinteger("Input", "Enter number of levels:")
            if levels is None:
                return
        else:

            return

        # Perform quantization
        continuous_indices, continuous_original, continuous_quantized = quantize_signal(signal_indices, signal_samples,
                                                                                        levels,islevel)

        # Plot both original and quantized signals
        plot_original_and_quantized(continuous_indices, continuous_original, continuous_quantized)
    else:
        messagebox.showwarning("No Signal", "Please load a signal first.")


def load2Signals():
    global signal_indices1, signal_samples1,signal_indices2, signal_samples2

    file_path = filedialog.askopenfilename(title="Select the 1st Signal File", filetypes=[("Text Files", "*.txt")])
    if file_path:
        try:
            signal_indices1, signal_samples1 = ReadSignalFile(file_path)
            if len(signal_indices1) > 0:



                file_path2 = filedialog.askopenfilename(title="Select the 2nd Signal File",
                                                           filetypes=[("Text Files", "*.txt")])
                if file_path2:
                    try:
                        signal_indices2, signal_samples2 = ReadSignalFile(file_path2)
                        if len(signal_indices2) > 0:
                            messagebox.showinfo("Signal ", f"Done")
                        else:
                            raise ValueError("Empty signal data.")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to display signal: {e}")
                        signal_label.config(text="No signal .")


            else:
                raise ValueError("Empty signal data.")
        except Exception as e:
            try:
                signal_samples1 = ReadFile(file_path)
                if len(signal_samples1) > 0:

                    file_path2 = filedialog.askopenfilename(title="Select the 2nd Signal File",
                                                            filetypes=[("Text Files", "*.txt")])
                    if file_path2:
                        try:
                            signal_samples2 = ReadFile(file_path2)
                            if len(signal_samples2) <= 0:

                                raise ValueError("Empty signal data.")
                        except Exception as e:
                            messagebox.showerror("Error", f"Failed to display signal: {e}")
                            signal_label.config(text="No signal .")


                else:
                    raise ValueError("Empty signal data.")
            except Exception as e:

                messagebox.showerror("Error", f"Failed to display signal: {e}")
                signal_label.config(text="No signal .")


    else:
        return


# Moving Average Function (without np.convolve)
def moving_average_gui():
    # Ask the user to input the signal manually
    load_signal()
    window_size = simpledialog.askinteger("Input", "Enter the window size:")
    if window_size is None or window_size <= 0:
        messagebox.showwarning("Invalid Input", "Please enter a positive integer.")
        return

            # Ensure the window size does not exceed the number of signal points
    if window_size > len(signal_samples):
        messagebox.showwarning("Invalid Input",
              "Window size cannot be greater than the number of signal points.")
        return

            # Calculate the moving average manually
    moving_avg = []
    for i in range(len(signal_samples) - window_size + 1):
        avg = sum(signal_samples[i:i+window_size]) / window_size
        moving_avg.append(avg)

            # Create new indices to match the moving average length
        new_indices = range(len(moving_avg))

            # Plot the moving average
    plot_signal(new_indices, moving_avg, title=f"Moving Average (Window Size: {window_size})")
    MovingAverage(window_size, new_indices, moving_avg)


def derivative_gui():

            load_signal()

            # Initialize an empty list for the first derivative
            first_derivative = []

            # Handle the middle elements with the formula: Y(n) = x(n) - x(n-1)
            for i in range(1, len(signal_samples)):
                first_derivative.append(signal_samples[i] - signal_samples[i - 1])

            # Create indices for plotting (skip the first index)
            derivative_indices1 = range(0, len(signal_samples)-1)

            # Plot the first derivative
            firstderivative(derivative_indices1,first_derivative)
            plot_signal(derivative_indices1, first_derivative, title="First Derivative of Signal")


            second_derivative = []

            # Handle the middle elements with the formula: Y(n) = x(n+1) - 2x(n) + x(n-1)
            for i in range(1, len(signal_samples) - 1):
               second_derivative.append(signal_samples[i + 1] - 2 * signal_samples[i] + signal_samples[i - 1])

            # Create indices for plotting (ignore the first and last index)
            derivative_indices2 = range(0, len(signal_samples) - 2)

            # Plot the second derivative
            secondderivative(derivative_indices2,second_derivative)
            plot_signal(derivative_indices2, second_derivative, title="Second Derivative of Signal")




def convolve_signals_gui():
    global signal_indices1, signal_samples1, signal_indices2, signal_samples2
    load2Signals()  # Function for loading two signals

    try:
        # Get the lengths of the input signals
        len1 = len(signal_samples1)
        len2 = len(signal_samples2)

        # Length of the resulting convolution signal
        len_conv = len1 + len2 - 1

        # Initialize the convolution result array with zeros
        y = np.zeros(len_conv)

        # Manual convolution
        for i in range(len1):
            for j in range(len2):
                y[i + j] += signal_samples1[i] * signal_samples2[j]

        # Calculate the starting index of the convolution result
        start_index = signal_indices1[0] + signal_indices2[0]

        # Create the new indices for the convolution result
        new_indices = np.arange(start_index, start_index + len_conv)

        # Plot the convolved signal
        plot_signal(new_indices, y, title="Convolution of Signals")
    except Exception as e:
        messagebox.showerror("Error", f"Convolution failed: {e}")
    convSignalSamplesAreEqual('Signal 1.txt', 'Signal 2.txt', new_indices, y)


# Function to compute DFT or IDFT
def compute_dft(samples, sampling_frequency, inverse=False):
    N = len(samples)

    amplitude=[]
    phase=[]
    transform_type = "IDFT" if inverse else "DFT"

    # Fourier Transform or Inverse Fourier Transform
    if inverse:
        result = np.fft.ifft(samples)  # Inverse
    else:

        for k in range(N):
            real = 0
            complex = 0
            for n in range(N):
                real+=signal_samples[n]*np.cos(2*np.pi*k*n/N)
                complex+=-signal_samples[n]*np.sin(2*np.pi*k*n/N)

            amp=np.sqrt(real*real+complex*complex)
            amplitude.append(amp)
            if(real>=0):
                phase.append(np.arctan(complex/real))
            elif(complex>=0):
                phase.append(np.arctan(complex/real)+np.pi)
            else:
                phase.append(np.arctan(complex / real) - np.pi)



    freqs=[]
    for i in range(1,N+1):
        freqs.append(i* 2*np.pi*sampling_frequency/N)

    # For FT: compute Amplitude and Phase
    if not inverse:
        ampout,phout=ReadSignalFile(r"C:\Users\User\Downloads\Lab2\Task 1 testcases and testing functions\Task 1 testcases and testing functions\Task 1 testcases and testing functions\DFT\Output_Signal_DFT_A,Phase.txt")
        SignalComapreAmplitude(amplitude,ampout)
        SignalComaprePhaseShift(phase,phout)
        return freqs, amplitude, phase
    # For IDFT: return reconstructed signal
    else:
        return np.real(result)

# Function to handle Fourier Transform
def handle_fourier_transform():
    global signal_indices, signal_samples
    file_name = signal_file_var.get()
    sampling_frequency = sampling_freq_var.get()

    if not file_name or not sampling_frequency:
        messagebox.showerror("Error", "Please select a file and enter sampling frequency.")
        return

    try:
        sampling_frequency = float(sampling_frequency)
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid numeric value for sampling frequency.")
        return

    try:
        # Use the globally loaded signal
        signal_indices, signal_samples = ReadSignalFile(file_name)
        freqs, amplitude, phase = compute_dft(loaded_samples, sampling_frequency)

        # Plot Frequency vs Amplitude
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.stem(freqs, amplitude, 'b')
        plt.title('Frequency vs Amplitude')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid()

        # Plot Frequency vs Phase
        plt.subplot(2, 1, 2)
        plt.stem(freqs, phase, 'r')
        plt.title('Frequency vs Phase')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (radians)')
        plt.grid()

        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox


def compute_idft():
    # Browse and load the signal file
    file_path = signal_file_var.get()

    if not file_path:
        messagebox.showerror("Error", "No file selected.")
        return

    try:
        # Read signal data from the file
        expected_indices = []
        expected_samples = []

        with open(file_path, 'r') as f:
            lines = f.readlines()

            # Skip the first few lines if they are not part of the data (assuming first 3 lines are not useful)
            lines = lines[3:]  # Adjust this based on your file structure

            for line in lines:
                L = line.strip()
                if len(L.split()) == 1:  # Only amplitude, no phase shift
                    expected_indices.append(float(L))
                    expected_samples.append([float(L), 0])  # Assume phase shift is 0 if not provided
                elif len(L.split()) == 2:  # Amplitude and phase shift
                    amplitude, phase_shift = L.split()
                    expected_indices.append(float(amplitude))
                    expected_samples.append([float(amplitude), float(phase_shift)])
                else:
                    break  # Stop reading if the format is unexpected

        # Now, expected_samples should contain pairs [amplitude, phase_shift]
        loaded_samples = expected_samples

        # Extract the amplitudes and phase shifts from the loaded signal
        amplitudes = []
        phase_shifts = []

        for sample in loaded_samples:
            if len(sample) == 2:  # Ensure there are exactly two values: amplitude and phase shift
                amplitudes.append(sample[0])  # First value is amplitude
                phase_shifts.append(sample[1])  # Second value is phase shift
            else:
                messagebox.showerror("Error", "Invalid data format in signal file.")
                return

        # Now reconstruct the signal using the IDFT formula
        N = len(amplitudes)
        reconstructed_samples = []

        for n in range(N):
            real_sum = 0
            for k in range(N):
                # Create the complex number from amplitude and phase
                complex_component = amplitudes[k] * np.cos(phase_shifts[k]) + 1j * amplitudes[k] * np.sin(
                    phase_shifts[k])
                real_sum += complex_component * np.exp(1j * 2 * np.pi * k * n / N)  # Apply IDFT formula

            # Normalize the result
            reconstructed_samples.append(np.real(real_sum) / N)
        ind,val= ReadSignalFile(r"C:\Users\User\Downloads\Lab2\Task 1 testcases and testing functions\Task 1 testcases and testing functions\Task 1 testcases and testing functions\IDFT\Output_Signal_IDFT.txt")
        SignalComapreAmplitude(reconstructed_samples,val)
        # Plot the reconstructed signal
        plt.figure(figsize=(10, 6))
        plt.plot(reconstructed_samples, label="Reconstructed Signal", linestyle='-', marker='x')
        plt.title("Reconstructed Signal from Amplitudes and Phase Shifts")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during signal reconstruction: {str(e)}")


# Function to browse and select a file, then read using ReadSignalFile
def browse_file():
    file_path = filedialog.askopenfilename(
        title="Select a Signal File",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if file_path:
        signal_file_var.set(file_path)
        try:
            indices, samples = ReadSignalFile(file_path)
            loaded_indices_samples.set(f"Loaded {len(indices)} samples.")
            # Store the loaded signal in the global variables for later use
            global loaded_indices, loaded_samples
            loaded_indices, loaded_samples = indices, samples
        except Exception as e:
            messagebox.showerror("Error", f"Could not read the file: {e}")
            signal_file_var.set("")
            loaded_indices_samples.set("")

#project
def correlation():
    load2Signals()
    N=len(signal_samples1)
    R1=[]
    res=[]

    s1=0
    s2=0
    for j in range(N):
        r=0
        s1+=signal_samples1[j]*signal_samples1[j]
        s2+=signal_samples2[j]*signal_samples2[j]


        for n in range(N):
            r+=signal_samples1[n]*signal_samples2[(n+j)%N]



        R1.append(r/N)

    for j in range(N):
        res.append(R1[j]/(np.sqrt(s1*s2)/N))
    return res

def corr_gui():
    res=correlation()
    Compare_Signals("CorrOutput.txt",signal_indices1,res)
    plt.figure(figsize=(10, 6))
    plt.stem(signal_indices1, res, 'b')

    plt.xlabel('index')
    plt.ylabel('sample')
    plt.grid()



    plt.tight_layout()
    plt.show()
def Time_delay(res,Fs):



    max=-10
    ind=-1
    N=len(signal_samples1)
    for j in range(N):


        if (res[j]>max):
            max=res[j]
            ind=j
    result = ind/Fs
    print("Fs= ", Fs)
    print("Excpected output = ",result)
    return max

def Time_gui():
    res = correlation()

    Fs = simpledialog.askinteger("Input", "Enter the Fs :")
    if Fs is None or Fs <= 0:
        messagebox.showwarning("Invalid Input", "Please enter a positive integer.")
        return
    Time_delay(res, Fs)
def TemplateMatching():
    print("down1")
    down1=correlation()
    maxdown1= Time_delay(down1, 100)
    print("down2")
    down2 = correlation()
    maxdown2=Time_delay(down2, 100)
    print("down3")
    down3 = correlation()
    maxdown3=Time_delay(down3, 100)
    print("down4")
    down4 = correlation()
    maxdown4=Time_delay(down4, 100)
    print("down5")
    down5 = correlation()
    maxdown5=Time_delay(down5, 100)
    print("up1")
    up1=correlation()
    maxup1= Time_delay(up1,100)
    print("up2")
    up2 = correlation()
    maxup2=Time_delay(up2, 100)
    print("up3")
    up3 = correlation()
    maxup3=Time_delay(up3, 100)
    print("up4")
    up4 = correlation()
    maxup4=Time_delay(up4, 100)
    print("up5")
    up5 = correlation()
    maxup5=Time_delay(up5, 100)

    up=(maxup1+maxup2+maxup3+maxup4+maxup5)/5
    down=(maxdown1+maxdown2+maxdown3+maxdown4+maxdown5)/5

    if(up>down):
        print("signal is up")
    else:
        print("signal is down")

# Main GUI Window
root = tk.Tk()
root.title("Signal Processing GUI")
root.geometry("500x400")
root.config(bg="#f0f0f0")

# Create Notebook for tabs
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
notebook.add(task1_frame, text='Task 1')
notebook.add(task2_frame, text='Task2')

notebook.add(task3_frame, text='Task 3')
notebook.add(task5_frame, text='Task 5')
notebook.add(task6_tab, text="Task 6")
notebook.add(task7_tab, text="Task 7")
# Layout frames for Task 1
top_frame = tk.Frame(task1_frame, bg="#f0f0f0")
top_frame.pack(pady=10)

middle_frame = tk.Frame(task1_frame, bg="#f0f0f0")
middle_frame.pack(pady=10)

bottom_frame = tk.Frame(task1_frame, bg="#f0f0f0")
bottom_frame.pack(pady=10)

# Signal label
signal_label = tk.Label(top_frame, text="No signal loaded.", bg="#f0f0f0", font=("Helvetica", 12))
signal_label.pack(pady=10)


# Function to create and place buttons
def create_button(frame, text, command, row, column, width=20, bg="#4CAF50", fg="white"):
    button = tk.Button(frame, text=text, command=command, width=width, bg=bg, fg=fg)
    button.grid(row=row, column=column, padx=10, pady=5)
    return button


# Action buttons for Task 1
create_button(middle_frame, "Load Signal", load_signal, 0, 0, bg="#4CAF50")
create_button(middle_frame, "Plot Signal", plot_signal_gui, 0, 1, bg="#4CAF50")
create_button(middle_frame, "Amplify Signal", amplify_signal_gui, 1, 0, bg="#2196F3")
create_button(middle_frame, "Add Signal", add_signal_gui, 1, 1, bg="#2196F3")
create_button(middle_frame, "Subtract Signal", subtract_signal_gui, 2, 0, bg="#2196F3")
create_button(middle_frame, "Shift Signal", shift_signal_gui, 2, 1, bg="#FF9800")
create_button(middle_frame, "Fold Signal", fold_signal_gui, 3, 0, bg="#FF9800")

# Clear and Exit buttons for Task 1
create_button(bottom_frame, "Clear Signal", clear_signal, 0, 0, bg="#F44336")
create_button(bottom_frame, "Exit", root.quit, 1, 0, bg="#F44336")

# Signal Generation buttons for Task 2
create_button(task2_frame, "Generate Sine Wave", lambda: generate_signal_gui("sine"), 0, 0, bg="#4CAF50")
create_button(task2_frame, "Generate Cosine Wave", lambda: generate_signal_gui("cosine"), 0, 1, bg="#4CAF50")
create_button(task2_frame, "Display signal",  displaySignal, 1, 0, bg="#2196F3")
# Action buttons for Task 3
create_button(task3_frame, "Quantize Signal", quantize_signal_gui, 0, 0, bg="#4CAF50")

create_button(task5_frame, "Convolve Signals", convolve_signals_gui, 3, 0, bg="#FF9800")
create_button(task5_frame, "Moving Average", moving_average_gui, 0, 0, bg="#4CAF50")
create_button(task5_frame, " Derivative", derivative_gui, 1, 0, bg="#2196F3")

# Add input fields and button
tk.Label(task6_tab, text="Select Signal File:").pack(pady=5)
signal_file_var = tk.StringVar()
loaded_indices_samples = tk.StringVar()
tk.Entry(task6_tab, textvariable=signal_file_var, state="readonly", width=50).pack(pady=5)
tk.Button(task6_tab, text="Browse", command=browse_file).pack(pady=5)
tk.Label(task6_tab, textvariable=loaded_indices_samples, fg="green").pack(pady=5)

tk.Label(task6_tab, text="Sampling Frequency (Hz):").pack(pady=5)
sampling_freq_var = tk.StringVar()
tk.Entry(task6_tab, textvariable=sampling_freq_var).pack(pady=5)

tk.Button(task6_tab, text="Compute Fourier Transform", command=handle_fourier_transform).pack(pady=10)
tk.Button(task6_tab, text="Reconstruct Signal (IDFT)", command=compute_idft).pack(pady=10)

create_button(task7_tab, " Correlation", corr_gui, 1, 0, bg="#2196F3")
create_button(task7_tab, " Time delay", Time_gui, 1, 1, bg="#2196F3")
create_button(task7_tab, " TemplateMatching", TemplateMatching, 2, 0, bg="#2196F3")

# Initialize global variables for loaded signal
loaded_indices = []
loaded_samples = []
# Start the tkinter main loop
root.mainloop()