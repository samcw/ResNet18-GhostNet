import pandas as pd
import matplotlib.pyplot as plt

withoutGhost = []
withGhost = []
withAll = []
withGhostAndEca = []
epoch = []

for i in range(50):
    epoch.append(i + 1)

dataNotWithGhost = pd.read_csv('./withghost.csv', usecols=[0])
dataWithGhost = pd.read_csv('./withoutghost.csv', usecols=[0])
dataWithAll = pd.read_csv('./withall.csv', usecols=[0])
dataWithGhostAndEca = pd.read_csv('./withGhostEcanet.csv', usecols=[0])


for index, row in dataNotWithGhost.iterrows():
    withGhost.append(float(row['with']))

for index, row in dataWithGhost.iterrows():
    withoutGhost.append(float(row['with']))

for index, row in dataWithAll.iterrows():
    withAll.append(float(row['with']))

for index, row in dataWithGhostAndEca.iterrows():
    withGhostAndEca.append(float(row['with']))

print(withoutGhost)
print(withGhost)


plt.plot(epoch, withoutGhost, linewidth=0.5, color='b', label='withoutGhost')
# plt.plot(epoch, withGhost, linewidth=0.5, color='b', label='withGhost')
# plt.plot(epoch, withAll, linewidth=0.5, color='g', label='withAll')
plt.plot(epoch, withGhostAndEca, linewidth=0.5, color='r', label='withGhostAndEca')

plt.xlabel('Epochs')
plt.ylabel('Acc')

plt.title('Accuracy trend')
plt.legend()

plt.savefig('res.png')
plt.show()