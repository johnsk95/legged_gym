import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

d = pd.read_csv('gait_info.csv').to_numpy()

plt.plot(d[:,0])
plt.show()

