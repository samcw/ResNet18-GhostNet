import pandas as pd
import matplotlib.pyplot as plt

withoutGhost = []
withGhost = []
withSwish = []
epoch = []

for i in range(50):
    epoch.append(i + 1)

dataNotWithGhost = pd.read_csv('./with.csv', usecols=[0])
dataWithGhost = pd.read_csv('./without.csv', usecols=[0])
dataWithSwish = pd.read_csv('./withswish.csv', usecols=[0])

for index, row in dataNotWithGhost.iterrows():
    withoutGhost.append(int(row['with'][7:9]))

for index, row in dataWithGhost.iterrows():
    withGhost.append(int(row['without'][7:9]))

for index, row in dataWithSwish.iterrows():
    withSwish.append(int(row['with'][7:9]))

print(epoch)

plt.plot(epoch, withoutGhost, color='r', label='withoutGhost')
plt.plot(epoch, withGhost, color=(0, 0, 0), label='withGhost')
plt.plot(epoch, withSwish, color='b', label='withSwish')

plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.title('Accuracy trend')
plt.legend()

plt.savefig('res.png')
plt.show()