import numpy as np
import matplotlib.pyplot as plt
import os

def load_ground_truth_hr(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        with open(filepath, 'r') as f:
            lines = f.readlines()
        hr_line = lines[1]
        return [float(x) for x in hr_line.strip().split() if x]
    elif ext == ".xmp":
        hr_values = []
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip() or not any(char.isdigit() for char in line):
                    continue
                parts = line.replace(',', ' ').replace('\t', ' ').split()
                if len(parts) < 2:
                    continue
                try:
                    hr = float(parts[1])
                    hr_values.append(hr)
                except ValueError:
                    continue
        if not hr_values:
            raise ValueError("No HR values found in gtdump.xmp.")
        return hr_values
    else:
        raise ValueError(f"Unsupported ground truth file format: {ext}")

def analyze_ground_truth_vs_calculated(ground_truth_hr, calculated_hr):
    calc_arr = np.array(calculated_hr)
    gt_arr = np.array(ground_truth_hr[:len(calc_arr)])
    valid_idx = ~np.isnan(calc_arr)
    valid_calc = calc_arr[valid_idx]
    valid_gt = gt_arr[valid_idx]
    if len(valid_calc) > 0:
        mae = np.mean(np.abs(valid_gt - valid_calc))
        print(f"Aligned Ground Truth HR: {valid_gt}")
        print(f"Aligned Calculated HR: {valid_calc}")
        print(f"Mean Absolute Error (MAE) on valid frames: {mae:.2f}")
        try:
            plt.figure()
            plt.plot(valid_gt, label='Ground Truth HR')
            plt.plot(valid_calc, label='Calculated HR')
            plt.xlabel('Frame (valid only)')
            plt.ylabel('HR (bpm)')
            plt.legend()
            plt.title('HR Comparison (Valid Frames Only)')
            plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")
    else:
        print("No valid calculated HR values to compare.")
