import numpy as np
import matplotlib.pyplot as plt
import FunctionFile as FF


#variables
t = np.linspace(0, 17500)
J0 = 0.1 #m/s

#Parameters
Kb = 1e-3
Ki = 1e-4
Ks = 0.00021599272305759687
Kc = 0.000761947260408238

alpha = 0.005
base = 0.001
tf = 2018.8


Pot = FF.SigmoidCakeStandard(Ks, Kc, alpha, tf, base, J0, t)
#Pot = FF.CakeStandard(Kc, Ks, J0, t)
#Pot = FF.Standard(Ks, J0, t)

plt.figure(figsize=(10, 8))
#plt.subplot(2, 2, 1)
plt.plot(t, Pot)
plt.ylabel(r"$\frac{P}{P0}$", rotation=0, fontsize=20, labelpad=20)
plt.xlabel("t", fontsize=20)
plt.title("Complete Blocking", fontsize=25)
plt.show()