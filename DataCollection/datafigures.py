import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

font = {'weight' : 'normal',
        'size'   : 22}

axes = {'titlesize'  : 22,
        'labelsize'  : 22}

legend = {'fontsize'  : 22}

figure = {'figsize'  : (10,5)}

matplotlib.rc('font', **font)
matplotlib.rc('axes', **axes)
matplotlib.rc('legend', **legend)
matplotlib.rc('figure', **figure)

data = pd.read_csv('../data/clustereddata.csv')
# data = data.sample(frac=0.1, replace=False)

timedifference = []
for index, row in data.iterrows():
  timedifference.append(row['comment_utc'] - row['submission_utc'])

data['timedifference'] = timedifference

# we want to be able to graph expected comment score as a function of timedifference
# in a 20 hour period, take groups of 30 minutes of data and graph a single point (average for each)
newdata = []
for i in range(0, 39):
  tempdata = data.loc[(data['timedifference'] < 180000*(i+1)) & (data['timedifference'] > 180000*i)]
  newdata.append([i,tempdata['comment_score'].mean()])

print(newdata)

pdnewdata = pd.DataFrame(newdata, columns=list('xy'))

# plot the comment score versus time
plt.scatter(pdnewdata['x'], pdnewdata['y'])
plt.plot(pdnewdata['x'], pdnewdata['y'])
plt.title('Vote Decay Over Time')
plt.xlabel('Time Difference Between Submission and Comment')
plt.ylabel('Mean Score')
plt.show()
