import  subprocess
import numpy as np

incr = 0.02
bot = 0
top = 1


for i in np.linspace(bot+incr, top-incr, int((top - bot) / incr) - 1):
    print("Currently running A = " + str(i))
    subprocess.run(["python",  "-u", "AS_test.py",  "%.6f" % i])
