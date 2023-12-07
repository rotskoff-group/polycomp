import  subprocess
import numpy as np

incr = 0.02
top = 1.5


for i in np.linspace(incr, top-incr, int(top / incr) - 1):
    print("NUMBER OF SCAN IS %.2f" % i)
    subprocess.run(["python",  "-u", "CART1_uncharged_scan.py",  "%.2f" % i])
