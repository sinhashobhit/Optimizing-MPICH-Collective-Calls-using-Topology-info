#!/usr/bin/env python
# coding: utf-8
#To run python3 plot.py 
# In[47]:

import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
sns.set()


# In[48]:
exec_time = []
with open(sys.argv[1]) as file_in:
    for line in file_in:
        if(line!="\n"):
            exec_time.append(float(line))
demo_input_format = pd.DataFrame.from_dict({
    "D": [],
    "P": [],
    "ppn": [],
    "mode": [],  # 1 --> optimized, 0 --> standard
    "time": [],
})

speedup_format = pd.DataFrame.from_dict({
    "D": [],
    "P": [],
    "ppn": [],
    "speed": [],
})


ctr = 0
for execution in range(10):
    for P in [4, 16]:
        for ppn in [1, 8]:
            for D in [16, 256, 2048]:
                # Change with the actual data
                demo_input_format = demo_input_format.append({
                    "D": D, "P": P, "ppn": ppn, "mode": 0, "time": exec_time[ctr]
                }, ignore_index=True)
                ctr = ctr+1
                
                demo_input_format = demo_input_format.append({
                    "D": D, "P": P, "ppn": ppn, "mode": 1, "time": exec_time[ctr]
                }, ignore_index=True)
                ctr = ctr+1
                speedup_format = speedup_format.append({
                    "D": D, "P": P, "ppn": ppn, "speed":exec_time[ctr-2]/exec_time[ctr-1]
                }, ignore_index=True)

demo_input_format["(P, ppn)"] = list(map(lambda x, y: ("(" + x + ", " + y + ")"), map(str, demo_input_format["P"]), map(str, demo_input_format["ppn"])))
'''
#To generate Statistics on Speedups
#print(demo_input_format)
for P in [4, 16]:
    for ppn in [1, 8]:
        for D in [16, 256, 2048]:
            temp = (speedup_format.loc[(speedup_format['P'] == P) & (speedup_format['ppn'] == ppn) & (speedup_format['D'] == D)])
            temp = temp.aggregate({"speed":['min','max','mean','median']})
            stat_list = temp["speed"].tolist()
            print("P=",P,", ppn=",ppn,", D=",D,"[Median Speedup = "+"{:.2f}".format(stat_list[3])+"]")

      
            #To generate Latex Tables
            #print(P,"&",ppn,"&",D,"& ","{:.2f}".format(stat_list[0]),"&","{:.2f}".format(stat_list[1]),"&","{:.2f}".format(stat_list[2]),"&","{:.2f}".format(stat_list[3]),"\\\\")
            print("\\hline")
'''         

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
##print(demo_input_format)


g = sns.catplot(x="(P, ppn)", y="time", data=demo_input_format, kind="box", col="D", hue="mode",showfliers=False)
g.fig.get_axes()[0].set_yscale('log')
g.set(xlabel='(P, ppn)', ylabel='Time (logarithmic scale) in s')
# name =  sys.argv[1]
# func_name = name[4:-4]
# print(func_name) 
# plt.title("MPI_"+func_name+" Vs Optimized MPI_"+func_name)
#plt.show()
plt.savefig(sys.argv[2])