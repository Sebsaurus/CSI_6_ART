import matplotlib.pyplot as plt

x=list(range(100)) 
y=[i**2 for i in x]

plt.plot(x,y)

plt.show()