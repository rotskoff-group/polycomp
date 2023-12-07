import  subprocess
import numpy as np

incr = 0.02
top = 1


for i in np.linspace(incr, top-incr, int(top / incr) - 1):
    subprocess.run(["python",  "AB_test.py",  "%.2f" % i])
