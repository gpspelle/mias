import pandas as pd
import matplotlib.pyplot as plt
filename = "truth.csv"
truth = pd.read_csv(filename)

truth['severity'].value_counts().plot(kind='pie')
plt.show()
#plot = truth.plot.pie(y='class', figsize=(5, 5))
#plot = truth.plot.pie(y='severity', figsize=(5, 5))
