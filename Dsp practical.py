import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import ttk, Menu, filedialog


def Compare_Signals(file_name, Your_indices, Your_samples):
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            print("Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one")
            return
    print("Test case passed successfully")


def ReadSignalFile(filename):
    expected_indices = []
    expected_samples = []
    with open(filename, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = float(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    return expected_indices, expected_samples


# --- Helper Functions ---
def calculate_filter_order(transition_band, fs, stop_band):
    """Calculate the FIR filter order based on transition band."""
    delta_f = transition_band / fs  # Normalize the transition band
    if (stop_band <= 21):
        N = 0.9 / delta_f  # Based on given formula
    elif (stop_band <= 44):
        N = 3.1 / delta_f  # Based on given formula
    elif (stop_band <= 53):
        N = 3.3 / delta_f  # Based on given formula
    elif (stop_band <= 74):
        N = 5.5 / delta_f  # Based on given formula
    N = int(np.ceil(N))  # Ensure the order is an integer
    if N % 2 == 0:
        N += 1  # Ensure it is odd for symmetry
    return N


# --- Window Functions ---
def rectangular_window(N):
    """Rectangular window function."""
    # print("rectangular\n")
    return np.ones(N + 1)


def hanning_window(N):
    """Hamming window function."""
    # print(" hanning\n")
    window = np.zeros(N + 1)
    start_idx = -(N - 1) // 2
    end_idx = (N - 1) // 2
    # Compute the window values using the given formula in a loop
    for n in range(start_idx, end_idx + 1):
        window[n - start_idx] = 0.5 + 0.5 * np.cos(2 * np.pi * n / N)
    return window


def hamming_window(N):
    """Hamming window function."""
    # print("hamming\n")
    window = np.zeros(N + 1)
    start_idx = -(N - 1) // 2
    end_idx = (N - 1) // 2
    # Compute the window values using the given formula in a loop
    for n in range(start_idx, end_idx + 1):
        window[n - start_idx] = 0.54 + 0.46 * np.cos(2 * np.pi * n / N)
    return window


def blackman_window(N):
    """Blackman window function."""
    print("blackman\n")
    window = np.zeros(N + 1)
    start_idx = -(N - 1) // 2
    end_idx = (N - 1) // 2
    # Compute the window values using the given formula in a loop
    for n in range(start_idx, end_idx + 1):
        window[n - start_idx] = 0.42 + 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    return window


def select_window_function(stopband, N):
    """Select window type based on stopband value."""
    if stopband <= 21:
        window = rectangular_window(N)
    elif stopband <= 44:
        window = hanning_window(N)
    elif stopband <= 53:
        window = hamming_window(N)
    elif stopband <= 74:
        window = blackman_window(N)
    else:
        raise ValueError("Stopband value out of supported range.")

    return window


# --- Ideal Impulse Response Functions ---
def ideal_impulse_response_lowpass(N, fc_adjusted, fs):
    """Generate lowpass ideal impulse response using start and end indexing."""
    start_idx = -(N - 1) // 2
    end_idx = (N - 1) // 2
    omega_c = 2 * np.pi * fc_adjusted  # Normalized angular frequency
    h = np.zeros(N + 1)
    # print("fc_adjusted   ", fc_adjusted)
    for idx in range(start_idx, end_idx + 1):
        if idx == 0:
            h[idx - start_idx] = 2 * fc_adjusted
        # print("sad")
        else:
            h[idx - start_idx] = 2 * fc_adjusted * np.sin(idx * omega_c) / (idx * omega_c)
        # print("h[idx - start_idx]   ",h[idx - start_idx])
        # print("indexxx   ",idx - start_idx)
    return h


def ideal_impulse_response_highpass(N, transition_band, fs):
    """Generate lowpass ideal impulse response using start and end indexing."""
    start_idx = -(N - 1) // 2
    end_idx = (N - 1) // 2
    fc_adjusted = (fc - (transition_band / 2)) / fs
    omega_c = 2 * np.pi * fc_adjusted  # Normalized angular frequency
    h = np.zeros(N + 1)
    # print("fc_adjusted   ", fc_adjusted)
    for idx in range(start_idx, end_idx + 1):
        if idx == 0:
            h[idx - start_idx] = 1 - 2 * fc_adjusted
        # print("sad")
        else:
            h[idx - start_idx] = -2 * fc_adjusted * np.sin(idx * omega_c) / (idx * omega_c)
    # print("h[idx - start_idx]   ",h[idx - start_idx])
    # print("indexxx   ",idx - start_idx)
    return h


def ideal_impulse_response_bandpass(N, f1, f2, fs):
    # print("N ",N)
    start_idx = -(N - 1) // 2
    end_idx = (N - 1) // 2
    omega2 = 2 * np.pi * (f2)
    omega1 = 2 * np.pi * (f1)
    h = np.zeros(N + 1)
    for idx in range(start_idx, end_idx + 1):
        if idx == 0:
            h[idx - start_idx] = 2 * (f2 - f1)
        else:
            h[idx - start_idx] = (
                    2 * f2 * np.sin(idx * omega2) / (idx * omega2)
                    - 2 * f1 * np.sin(idx * omega1) / (idx * omega1)
            )
    #  print("indexxx   ", idx - start_idx)
    # print("h[idx - start_idx]   ", h[idx - start_idx])
    return h


def ideal_impulse_response_bandstop(N, f1, f2, fs):
    start_idx = -(N - 1) // 2
    end_idx = (N - 1) // 2
    omega2 = 2 * np.pi * (f2)
    omega1 = 2 * np.pi * (f1)
    h = np.zeros(N + 1)
    for idx in range(start_idx, end_idx + 1):
        if idx == 0:
            h[idx - start_idx] = 1 - 2 * (f2 - f1)
        else:
            h[idx - start_idx] = (
                    2 * f1 * np.sin(idx * omega1) / (idx * omega1)
                    - 2 * f2 * np.sin(idx * omega2) / (idx * omega2)
            )
    return h


# --- Filter Design Logic ---
def design_filter(filter_type, fs, fc_adjusted, transition_band, stopband_atten):
    """Design the filter using windowed-sinc method."""
    try:
        # Calculate filter order
        N = calculate_filter_order(transition_band, fs, stopband_atten)
        print(f"Calculated Filter Order (N): {N}")

        # Generate base coefficients based on filter type
        if filter_type == "Lowpass":
            h = ideal_impulse_response_lowpass(N, fc_adjusted, fs)
        elif filter_type == "Highpass":
            h = ideal_impulse_response_highpass(N, transition_band, fs)
        else:
            raise ValueError("Currently only Lowpass is supported.")

        # Select the window based on stopband values
        window = select_window_function(stopband_atten, N)

        # Apply the window to the impulse response
        h_windowed = h * window

        print("Filter coefficients calculated.")
        return h_windowed, N
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def design_filter2(filter_type, fs, f1, f2, transition_band, stopband_atten):
    """Design the filter using windowed-sinc method."""
    try:
        # Calculate filter order
        N = calculate_filter_order(transition_band, fs, stopband_atten)
        print(f"Calculated Filter Order (N): {N}")

        # Generate base coefficients based on filter type
        if filter_type == "Bandpass":
            h = ideal_impulse_response_bandpass(N, f1, f2, fs)
        elif filter_type == "Bandstop":
            h = ideal_impulse_response_bandstop(N, f1, f2, fs)
        else:
            raise ValueError("Currently only Lowpass is supported.")

        # Select the window based on stopband values
        window = select_window_function(stopband_atten, N)

        # Apply the window to the impulse response
        h_windowed = h * window

        print("Filter coefficients calculated.")
        return h_windowed, N
    except Exception as e:
        print(f"Error: {e}")
        return None, None


# --- GUI Logic ---
def apply_filter():
    global fc

    # Fetch user input
    filter_type = filter_type_combo.get()
    fs = float(fs_entry.get())
    stopband_atten = float(stopband_atten_entry.get())
    transition_band = float(transition_band_entry.get())

    if filter_type in ["Lowpass", "Highpass"]:
        fc = float(fc_entry.get())

        # Adjust fc using the formula: fc_adjusted = fc + (transition_band / 2)
        if filter_type == "Lowpass":
            fc_adjusted = (fc + (transition_band / 2)) / fs
            # print(f"Adjusted cutoff frequency: {fc_adjusted}")
        else:
            fc_adjusted = (fc - (transition_band / 2)) / fs
            # print(f"Adjusted cutoff frequency: {fc_adjusted}")

        # Design the filter
        coeffs, N = design_filter(filter_type, fs, fc_adjusted, transition_band, stopband_atten)
        if coeffs is None:
            return
    elif filter_type in ["Bandpass", "Bandstop"]:
        f1 = float(f1_entry.get())
        f2 = float(f2_entry.get())
        if filter_type == "Bandpass":
            f1_adjusted = (f1 - (transition_band / 2)) / fs
            f2_adjusted = (f2 + (transition_band / 2)) / fs
            # print(f"Adjusted cutoff frequency: {f1_adjusted}")
            # print(f"Adjusted cutoff frequency: {f2_adjusted}")
        else:
            f1_adjusted = (f1 + (transition_band / 2)) / fs
            f2_adjusted = (f2 - (transition_band / 2)) / fs
            # print(f"Adjusted cutoff frequency: {f1_adjusted}")
            # print(f"Adjusted cutoff frequency: {f2_adjusted}")
        coeffs, N = design_filter2(filter_type, fs, f1_adjusted, f2_adjusted, transition_band, stopband_atten)
    else:
        raise ValueError("Invalid filter type selected.")
    # Generate input signal for visualization
    t = np.linspace(0, 1, int(fs), endpoint=False)
    input_signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(len(t))

    # Filter the signal using convolution
    filtered_signal = np.convolve(input_signal, coeffs, mode="same")

    # # Print coefficients in the requested format
    # print("0")
    # print("0")
    # print(N)
    start_idx = -(N-1) // 2  # Start index based on N
    # print("start ", start_idx)
    end_idx = (N-1) // 2  # End index based on N
    indexArr = []
    with open("filter_coefficients.txt", "w") as f:
        f.write("0\n0\n")
        f.write(f"{N}\n")
        for idx in range(start_idx, end_idx + 1):
            # Map index to the correct coefficient
            coeff_value = coeffs[idx - start_idx]
            indexArr.append(idx)
            # Write to file
            f.write(f"{idx} {coeff_value}\n")

            # Print to console
            # print(f"Index: {idx}, Coefficient: {coeff_value:.10f}")
    if filter_type == "Bandpass":
        Compare_Signals('BPFCoefficients.txt', indexArr, coeffs)
    elif filter_type == "Bandstop":
        Compare_Signals('BSFCoefficients.txt', indexArr, coeffs)
    elif filter_type == "Highpass":
        Compare_Signals('HPFCoefficients.txt', indexArr, coeffs)
    elif filter_type == "Lowpass":
        Compare_Signals('LPFCoefficients.txt', indexArr, coeffs)
    # Plot the original vs filtered signals
    plt.figure(figsize=(10, 6))
    plt.plot(t, input_signal, label="Original Signal")
    plt.plot(t, filtered_signal, label="Filtered Signal", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title(f"{filter_type} Filter")
    plt.grid()
    plt.show()

    messagebox.showinfo("Success", "Filter coefficients saved to filter_coefficients.txt")


def on_filter_type_change(event):
    filter_type = filter_type_combo.get()
    if filter_type in ["Bandpass", "Bandstop"]:
        fc_label.grid_remove()
        fc_entry.grid_remove()
        f1_label.grid(row=4, column=0, sticky=tk.W, pady=5)
        f1_entry.grid(row=4, column=1, sticky=tk.W, pady=5)
        f2_label.grid(row=5, column=0, sticky=tk.W, pady=5)
        f2_entry.grid(row=5, column=1, sticky=tk.W, pady=5)
    else:
        f1_label.grid_remove()
        f1_entry.grid_remove()
        f2_label.grid_remove()
        f2_entry.grid_remove()
        fc_label.grid(row=4, column=0, sticky=tk.W, pady=5)
        fc_entry.grid(row=4, column=1, sticky=tk.W, pady=5)


def convolve_two_signals(signal1, signal2):
    x1, y1 = signal1[0], signal1[1]
    x2, y2 = signal2[0], signal2[1]
    min_index = int(x1[0]) + int(x2[0])
    max_index = int(x1[-1]) + int(x2[-1])
    output_length = max_index - min_index + 1

    output_indices = list(range(min_index, max_index + 1))
    output_values = [0] * output_length

    #   y[n] = x[k] * h[n-k]
    for n in range(len(output_indices)):
        current_time = min_index + n
        sum = 0
        for k, (time1, val1) in enumerate(zip(x1, y1)):
            required_time = current_time - time1
            if required_time in x2:
                idx2 = x2.index(required_time)
                sum += val1 * y2[idx2]
        output_values[n] = sum
    return output_values, output_indices


def dft(signal_values):
    # print(signal_values)
    N = len(signal_values)
    dft_result = []
    for k in range(N):
        real = 0
        imag = 0
        for n in range(N):
            angle = -2 * np.pi * k * n / N
            real += np.double(signal_values[n]) * np.cos(angle)
            imag += np.double(signal_values[n]) * np.sin(angle)

        dft_result.append(complex(real, imag))
    # print("##########################")
    # print(dft_result)
    return np.array(dft_result)


def idft(dft_values):
    N = len(dft_values)
    idft_result = []
    for n in range(N):
        sum_complex = 0
        for k in range(N):
            angle = 2 * np.pi * k * n / N
            sum_complex += dft_values[k] * np.exp(1j * angle)
        idft_result.append(sum_complex / N)
    return np.array(idft_result)


def zero_pad(signal, length):
    """Zero pad a signal to the specified length."""
    return signal + [0] * (length - len(signal))


def convolve_fast(signal1, signal2):
    """Fast convolution using manually computed DFT and IDFT."""
    x1, y1 = signal1[0], signal1[1]
    x2, y2 = signal2[0], signal2[1]
    min_index = int(x1[0]) + int(x2[0])
    max_index = int(x1[-1]) + int(x2[-1])
    N = max_index - min_index + 1

    output_indices = list(range(min_index, max_index + 1))

    # Zero-pad both signals
    signal1_padded = zero_pad(y1, N)
    signal2_padded = zero_pad(y2, N)
    # print(signal1_padded)
    # Compute DFT of both signals
    dft_signal1 = dft(signal1_padded)
    dft_signal2 = dft(signal2_padded)

    # Point-wise multiplication in frequency domain
    dft_product = dft_signal1 * dft_signal2

    # Compute IDFT of the product
    convolved_signal = idft(dft_product)

    return convolved_signal.real, output_indices  # Return only the real part


signal_indices = []
signal_samples = []


def load_signal():
    global signal_indices, signal_samples
    file_path = filedialog.askopenfilename(title="Select a Signal File", filetypes=[("Text Files", "*.txt")])
    if file_path:
        try:
            signal_indices, signal_samples = ReadSignalFile(file_path)
            if len(signal_indices) > 0:
                # signal_label.config(text=f"Signal Loaded: {file_path.split('/')[-1]} (Length: {len(signal_samples)})")
                messagebox.showinfo("Signal Loaded", f"Signal loaded from {file_path}")
            else:
                raise ValueError("Empty signal data.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load signal: {e}")
        # signal_label.config(text="No signal loaded.")


def readTwoSignals_direct():
    inputFile = "filter_coefficients.txt"
    index_coff, sample_coff = ReadSignalFile(inputFile)

    signal1 = (index_coff, sample_coff)

    signal2 = (signal_indices, signal_samples)
    resVal, resIndex = convolve_two_signals(signal2, signal1)
    filter_type = filter_type_combo.get()
    if filter_type == "Bandpass":
        Compare_Signals('ecg_band_pass_filtered.txt', resIndex, resVal)
    elif filter_type == "Bandstop":
        Compare_Signals('ecg_band_stop_filtered.txt', resIndex, resVal)
    elif filter_type == "Highpass":
        Compare_Signals('ecg_high_pass_filtered.txt', resIndex, resVal)
    elif filter_type == "Lowpass":
        Compare_Signals('ecg_low_pass_filtered.txt', resIndex, resVal)
    plt.figure(figsize=(10, 6))
    plt.plot(resIndex, resVal, label="Convolve direct")

    plt.xlabel("index")
    plt.ylabel("value")
    plt.legend()
    plt.grid()
    plt.show()


def readTwoSignals_fast():
    inputFile = "filter_coefficients.txt"
    index_coff, sample_coff = ReadSignalFile(inputFile)
    signal1 = (index_coff, sample_coff)

    signal2 = (signal_indices, signal_samples)
    resVal, resIndex = convolve_fast(signal2, signal1)
    filter_type = filter_type_combo.get()
    if filter_type == "Bandpass":
        Compare_Signals('ecg_band_pass_filtered.txt', resIndex, resVal)
    elif filter_type == "Bandstop":
        Compare_Signals('ecg_band_stop_filtered.txt', resIndex, resVal)
    elif filter_type == "Highpass":
        Compare_Signals('ecg_high_pass_filtered.txt', resIndex, resVal)
    elif filter_type == "Lowpass":
        Compare_Signals('ecg_low_pass_filtered.txt', resIndex, resVal)
    plt.figure(figsize=(10, 6))
    plt.plot(resIndex, resVal, label="Convolve fast")

    plt.xlabel("index")
    plt.ylabel("value")
    plt.legend()
    plt.grid()
    plt.show()


# --- GUI setup ---
root = tk.Tk()
root.title("FIR Filter Designer")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

filter_type_label = ttk.Label(frame, text="Filter Type:")
filter_type_label.grid(row=0, column=0, sticky=tk.W, pady=5)
filter_type_combo = ttk.Combobox(frame, values=["Lowpass", "Highpass", "Bandpass", "Bandstop"], state="readonly")
filter_type_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
filter_type_combo.set("Lowpass")
filter_type_combo.bind("<<ComboboxSelected>>", on_filter_type_change)

fs_label = ttk.Label(frame, text="Sampling Frequency (Hz):")
fs_label.grid(row=1, column=0, sticky=tk.W, pady=5)
fs_entry = ttk.Entry(frame)
fs_entry.grid(row=1, column=1, sticky=tk.W, pady=5)

stopband_atten_label = ttk.Label(frame, text="Stopband Attenuation (dB):")
stopband_atten_label.grid(row=2, column=0, sticky=tk.W, pady=5)
stopband_atten_entry = ttk.Entry(frame)
stopband_atten_entry.grid(row=2, column=1, sticky=tk.W, pady=5)

transition_band_label = ttk.Label(frame, text="Transition Band (Hz):")
transition_band_label.grid(row=3, column=0, sticky=tk.W, pady=5)
transition_band_entry = ttk.Entry(frame)
transition_band_entry.grid(row=3, column=1, sticky=tk.W, pady=5)

fc_label = ttk.Label(frame, text="Cutoff Frequency (fc in Hz):")
fc_entry = ttk.Entry(frame)

f1_label = ttk.Label(frame, text="Lower Cutoff Frequency (f1 in Hz):")
f1_entry = ttk.Entry(frame)

f2_label = ttk.Label(frame, text="Upper Cutoff Frequency (f2 in Hz):")
f2_entry = ttk.Entry(frame)

apply_button = ttk.Button(frame, text="Apply Filter", command=apply_filter)
apply_button.grid(row=6, column=0, columnspan=2, pady=10)


#########load#####
# Function to create and place buttons
def create_button(frame, text, command, row, column, width=12, bg="#4CAF50", fg="white"):
    button = tk.Button(frame, text=text, command=command, width=width, bg=bg, fg=fg)
    button.grid(row=row, column=column, padx=10, pady=5)
    return button


create_button(frame, "Load Signal", load_signal, 6, 2, bg="#4CAF50")

create_button(frame, "convolve direct", readTwoSignals_direct, 6, 3, bg="#4CAF50")
create_button(frame, "convolve fast", readTwoSignals_fast, 6, 4, bg="#4CAF50")

on_filter_type_change(None)
root.mainloop()