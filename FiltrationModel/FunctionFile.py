import numpy as np

def sigmoid(t, tf, alpha):
    # Avoid overflow by using a piecewise function
    exponent = -alpha * (t - tf)
    safe_exp = np.where(exponent > 0, 1 / (1 + np.exp(-exponent)), np.exp(exponent) / (1 + np.exp(exponent)))
    return safe_exp

def Complete(Kb, J0, time_range):
    pressures = []
    for t in time_range:
        P_ratio = 1/(1-Kb*t)
        pressures.append(P_ratio)
    return pressures

def Standard(Ks, J0, time_range):
    pressures = []
    for t in time_range:
        P_ratio = (1+(Ks*J0*t/2))**2
        pressures.append(P_ratio)
    return pressures

def Intermediate(Ki, J0, time_range):
    pressures = []
    for t in time_range:
        P_ratio =np.exp(Ki*J0*t)
        pressures.append(P_ratio)
    return pressures

def Cake(Kc, J0, time_range):
    pressures = []
    for t in time_range:
        P_ratio =(1+2*Kc*(J0**2)*t)**0.5
        pressures.append(P_ratio)
    return pressures

def CakeComplete(Kc, Kb, J0, time_range):
    pressures = []
    for t in time_range:
        #P_ratio = (1/(1-Kb*t))*(1-((Kc*J0**2)/(Kb))*np.log(1-Kb*t))
        P_ratio = ((1+2*Kc*(J0**2)*t)**0.5)*(1/(1-Kb*t))
        pressures.append(P_ratio)
    return pressures

def CakeIntermediate(Kc, Ki, J0, time_range):
    pressures = []
    for t in time_range:
        #P_ratio = np.exp(Ki*J0*t)*(1+((Kc*J0)/(Ki))*(np.exp(Ki*J0*t)-1))
        P_ratio = ((1+2*Kc*(J0**2)*t)**0.5)*(np.exp(Ki*J0*t))
        pressures.append(P_ratio)
    return pressures

def CompleteStandard(Kb, Ks, J0, time_range):
    pressures = []
    for t in time_range:
        #P_ratio = 1/((1-Kb*t)*(1+((Ks*J0)/(2*Kb))*np.log(1-Kb*t))**2)
        P_ratio = 1/(1-Kb*t)*((1+(Ks*J0*t/2))**2)
        pressures.append(P_ratio)
    return pressures

def IntermediateStandard(Ki, Ks, J0, time_range):
    pressures = []
    for t in time_range:
        #P_ratio = (np.exp(Ki*J0*t))/((1-((Ks)/(2*Ki))*(np.exp(Ki*J0*t)-1))**2)
        P_ratio = (np.exp(Ki * J0 * t))*((1+(Ks*J0*t/2))**2)
        pressures.append(P_ratio)
    return pressures

def CakeStandard(Kc, Ks, J0, time_range):
    pressures = []
    for t in time_range:
        #P_ratio = ((1 - ((Ks * J0 * t) / 2)) ** (-2) + Kc * J0**2 * t)
        P_ratio = ((1+2*Kc*(J0**2)*t)**0.5)*((1 + (Ks * J0 * t / 2)) ** 2)
        pressures.append(P_ratio)
    return pressures

# Modified Sigmoidal version of CakeComplete
def SigmoidCakeComplete(Kc, Kb, alpha, tf, b, J0, time_range):
    pressures = []
    for t in time_range:
        # Time-dependent Kc and Kb
        Kc_t = Kc * (1-sigmoid(t, tf, alpha))  # Cake term dominates over time
        Kb_t = Kb * (sigmoid(t, tf, alpha))  # Standard term fades over time

        b_t = b * (1-sigmoid(t, tf, alpha))  # Cake term dominates over time
        # Pressure ratio calculation
        P_ratio = (((1 + 2 * Kc_t * (J0 ** 2) * t) ** 0.5) + b_t) * (1 / (1 - Kb_t * t))
        pressures.append(P_ratio)
    return pressures

# Modified Sigmoidal version of CakeIntermediate
def SigmoidCakeIntermediate(Kc, Ki, alpha, tf, b, J0, time_range):
    pressures = []
    for t in time_range:
        # Time-dependent Kc and Ki
        Kc_t = Kc * (1-sigmoid(t, tf, alpha))  # Cake term dominates over time
        Ki_t = Ki * (sigmoid(t, tf, alpha))  # Standard term fades over time

        b_t = b * (1-sigmoid(t, tf, alpha))  # Cake term dominates over time
        # Pressure ratio calculation
        P_ratio = (((1 + 2 * Kc_t * (J0 ** 2) * t) ** 0.5) + b_t) * (np.exp(Ki_t * J0 * t))
        pressures.append(P_ratio)
    return pressures

# Modified Sigmoidal version of CompleteStandard
def SigmoidCompleteStandard(Kb, Ks, alpha, tf, b, J0, time_range):
    pressures = []
    for t in time_range:
        # Time-dependent Kb and Ks
        Kb_t = Kb * (sigmoid(t, tf, alpha))  # Cake term fades over time
        Ks_t = Ks * (1 - sigmoid(t, tf, alpha))  # Standard term dominates over time

        b_t = b * (1-sigmoid(t, tf, alpha))  # Cake term dominates over time
        # Pressure ratio calculation
        P_ratio = 1 / (1 - Kb_t * t) * (((1 + (Ks_t * J0 * t / 2)) ** 2) + b_t)
        pressures.append(P_ratio)
    return pressures

# Modified Sigmoidal version of IntermediateStandard
def SigmoidIntermediateStandard(Ki, Ks, alpha, tf, b, J0, time_range):
    pressures = []
    for t in time_range:
        # Time-dependent Ki and Ks
        Ki_t = Ki * ( sigmoid(t, tf, alpha))  # Cake term fades over time
        Ks_t = Ks * (1 - sigmoid(t, tf, alpha))  # Standard term dominates over time

        b_t = b * (1-sigmoid(t, tf, alpha))  # Cake term dominates over time
        # Pressure ratio calculation
        P_ratio = (np.exp(Ki_t * J0 * t)) * (((1 + (Ks_t * J0 * t / 2)) ** 2) + b_t)
        pressures.append(P_ratio)
    return pressures

# Modified Sigmoidal version of CakeStandard
def SigmoidCakeStandard(Kc, Ks, alpha, tf, b,  J0, time_range):
    pressures = []
    for t in time_range:
        # Time-dependent Kc and Ks
        Kc_t = Kc * (1-sigmoid(t, tf, alpha))  # Cake term dominates over time
        Ks_t = Ks * ( sigmoid(t, tf, alpha))  # Standard term fades over time

        b_t = b * (1-sigmoid(t, tf, alpha))  # Cake term dominates over time

        # Pressure ratio calculation
        P_ratio = (((1 + 2 * Kc_t * (J0 ** 2) * t) ** 0.5) + b_t) * ((1 + (Ks_t * J0 * t / 2)) ** 2)
        pressures.append(P_ratio)
    return pressures
