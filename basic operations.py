import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline is used in Jupyter Notebook, remove it for other environments
# 1. Array operations
array = np.array([1, 2, 3, 4, 5])
squared_array = np.square(array)
mean_value = np.mean(array)
print("Original Array:", array)
print("Squared Array:", squared_array)
print("Mean Value:", mean_value)

# 2. DataFrame operations
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [24, 27, 22, 32],
    'Score': [85, 88, 92, 79]
}
df = pd.DataFrame(data)
print(df)

mean_age = df['Age'].mean()
print("Mean Age:", mean_age)

filtered_data = df[df['Score'] > 85]
print(filtered_data)

# 3. Plotting a sine wave
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Sine Wave')
plt.title('Sine Function')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.show()
