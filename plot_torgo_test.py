import json
import soundfile as sf
import os
import pandas
import matplotlib.pyplot as plt
import numpy as np
import sys
in_dict = pandas.read_csv(sys.argv[1])


plt.scatter(in_dict["correctness"],in_dict["predicted"],marker="x",alpha=0.5)
#plt.xlim(0,1)
#plt.ylim(0,1)

plt.xlabel("Correctness")
plt.ylabel("Predicted")
plt.title("Predicted vs Correctness(CER) for Speakers F01,M01")
plt.savefig(sys.argv[1].replace(".csv",".png"))
print("Spearman:",in_dict["correctness"].corr(in_dict["predicted"],method="spearman"))
print("Pearson:",in_dict["correctness"].corr(in_dict["predicted"],method="pearson"))