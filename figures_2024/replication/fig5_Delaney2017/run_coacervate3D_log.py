import  subprocess
import numpy as np
import sys

max_job = int(sys.argv[1])
curr_job = int(sys.argv[2])

subsamples = 2

bot = -5
top = 0.5

total_samples = (max_job + 1) * subsamples + 1
options = np.logspace(bot, top, num=total_samples)
print(options)
A_amt = options[curr_job]
for A_amt in options[curr_job * subsamples:(curr_job + 1) * subsamples]:
    print("NUMBER OF SCAN IS %.3e" % A_amt)
    subprocess.run(["python",  "-u", "coacervate_scan3D.py",  "%.10e" % A_amt])
