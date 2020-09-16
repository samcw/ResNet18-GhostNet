import pandas as pd
import matplotlib.pyplot as plt

withoutGhost = []
withGhost = []
withSwish = []
withSeNet = []
epoch = []

for i in range(50):
    epoch.append(i + 1)

dataNotWithGhost = pd.read_csv('./withghost.csv', usecols=[0])
dataWithGhost = pd.read_csv('./withoutghost.csv', usecols=[0])

for index, row in dataNotWithGhost.iterrows():
    withGhost.append(float(row['with']))

for index, row in dataNotWithGhost.iterrows():
    withoutGhost.append(float(row['with']))

print(withoutGhost)


plt.plot(epoch, withoutGhost, linewidth=0.5, color='r', label='withoutGhost')
plt.plot(epoch, withGhost,linewidth=0.5, color='b', label='withGhost')
# plt.plot(epoch, withSwish, color='b', label='withSwish')
# plt.plot(epoch, withSeNet, color='g', label='withSeNet')

plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.title('Accuracy trend')
plt.legend()

plt.savefig('res.png')
plt.show()