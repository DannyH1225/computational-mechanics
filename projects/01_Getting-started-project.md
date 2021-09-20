---
jupytext:
  formats: notebooks//ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```

# Computational Mechanics Project #01 - Heat Transfer in Forensic Science

We can use our current skillset for a macabre application. We can predict the time of death based upon the current temperature and change in temperature of a corpse. 

Forensic scientists use Newton's law of cooling to determine the time elapsed since the loss of life, 

$\frac{dT}{dt} = -K(T-T_a)$,

where $T$ is the current temperature, $T_a$ is the ambient temperature, $t$ is the elapsed time in hours, and $K$ is an empirical constant. 

Suppose the temperature of the corpse is 85$^o$F at 11:00 am. Then, 2 hours later the temperature is 74$^{o}$F. 

Assume ambient temperature is a constant 65$^{o}$F.

1. Use Python to calculate $K$ using a finite difference approximation, $\frac{dT}{dt} \approx \frac{T(t+\Delta t)-T(t)}{\Delta t}$.

```{code-cell} ipython3
T_f = 74
T_i = 85
T_a = 65
dt = 2
dT_dt = (T_f - T_i)/ dt

K = - dT_dt / (T_f - T_a)
print(f'K={K} per hour')
```

2. Change your work from problem 1 to create a function that accepts the temperature at two times, ambient temperature, and the time elapsed to return $K$.

```{code-cell} ipython3
def K(T_i, T_f, T_a, dt):
    '''Function that accepts the temperature at two times, ambient temperature, and the time elapsed to return 𝐾
    T_i = Initial temperature
    T_f = Final temperature
    T_a = Ambient temperature
    Governing equation: dT/dt = -K(T-T_a)
    '''
    dT_dt = (T_f - T_i) / dt
    K = - dT_dt / (T_f - T_a)
    return K
```

3. A first-order thermal system has the following analytical solution, 

    $T(t) =T_a+(T(0)-T_a)e^{-Kt}$

    where $T(0)$ is the temperature of the corpse at t=0 hours i.e. at the time of discovery and $T_a$ is a constant ambient temperature. 

    a. Show that an Euler integration converges to the analytical solution as the time step is decreased. Use the constant $K$ derived above and the initial temperature, T(0) = 85$^o$F. 

    b. What is the final temperature as t$\rightarrow\infty$?
    
    c. At what time was the corpse 98.6$^{o}$F? i.e. what was the time of death?

```{code-cell} ipython3
#part a
T_f = 74
T_i = 85
T_a = 65
dt = 2
dT_dt = (T_f - T_i)/ dt

K = - dT_dt / (T_f - T_a)
t_10 = np.linspace(0,5,10)
t_50 = np.linspace(0,5,50)


def cooling_law(T_i,T_a, K, t):
    T_f_ana = T_a + (T_i - T_a) * np.exp(-K*t)
    T_f_num = np.zeros(len(t))
    T_f_num[0] = T_i
    for i in range(1,len(t)):
        T_f_num[i] = -K*(T_f_num[i-1] - T_a)*(t[i]-t[i-1]) + T_f_num[i-1]
    
    return T_f_ana, T_f_num

T_10_ana, T_f10_num = cooling_law(T_i, T_a, K, t_10)

T_50_ana, T_f50_num = cooling_law(T_i, T_a, K, t_50)

plt.plot(t_10, T_f10_num, '-o', label= '10 time steps')
plt.plot(t_10, T_10_ana, '-', label= 'analytical 10 steps')
plt.title('Cooling for First 5 Hours')
plt.xlabel('time (hr)')
plt.ylabel('Temp \N{DEGREE SIGN}F')
plt.legend();
```

```{code-cell} ipython3
plt.plot(t_50, T_f50_num, '-o', label= '50 time steps')
plt.plot(t_50, T_50_ana, '-', label= 'analytical 50 steps')
plt.title('Cooling for First 5 Hours')
plt.xlabel('time (hr)')
plt.ylabel('Temp \N{DEGREE SIGN}F')
plt.legend();
```

Part a: Looking at the both graphs, it can be shown that as the time step decreases the Euler integration approaches the analytical solution.

+++

Part b: The final temperature as t--> infinity is the ambient temperature.

```{code-cell} ipython3
#part C
T_f = 74
T_i = 85
T_a = 65
dt = 2
dT_dt = (T_f - T_i)/ dt

K = - dT_dt / (T_f - T_a)
t_death = np.log((98.6 - T_a)/ (T_i - T_a)) / -K

print(t_death*60)
print('Time of death is 10:10 am')
```

4. Now that we have a working numerical model, we can look at the results if the
ambient temperature is not constant i.e. T_a=f(t). We can use the weather to improve our estimate for time of death. Consider the following Temperature for the day in question. 

    |time| Temp ($^o$F)|
    |---|---|
    |6am|50|
    |7am|51|
    |8am|55|
    |9am|60|
    |10am|65|
    |11am|70|
    |noon|75|
    |1pm|80|

    a. Create a function that returns the current temperature based upon the time (0 hours=11am, 65$^{o}$F) 
    *Plot the function $T_a$ vs time. Does it look correct? Is there a better way to get $T_a(t)$?

    b. Modify the Euler approximation solution to account for changes in temperature at each hour. 
    Compare the new nonlinear Euler approximation to the linear analytical model. 
    At what time was the corpse 98.6$^{o}$F? i.e. what was the time of death?

```{code-cell} ipython3
#part a

def ambient_temp(time):
    if 0<= time < 1:
        temp = 5*time + 70
        print(f'Ambient temp is {temp} \N{DEGREE SIGN}F')
    elif -1<= time < 0:
        temp = 5*time + 70
        print(f'Ambient temp is {temp} \N{DEGREE SIGN} F')
    elif -2<= time < -1:
        temp = 5*time + 70
        print(f'Ambient temp is {temp} \N{DEGREE SIGN} F')
    elif -3<= time < -2:
        temp = 5*time + 70
        print(f'Ambient temp is {temp} \N{DEGREE SIGN} F')
    elif -4<= time < -3:
        temp = 5*time + 70
        print(f'Ambient temp is {temp} \N{DEGREE SIGN} F')
    elif -5<= time < -5:
        print(f'Ambient temp is {temp} \N{DEGREE SIGN} F')
    elif 1<= time < 2:
        print(f'Ambient temp is {temp} \N{DEGREE SIGN} F')
    elif 2<= time:
        temp = 5*time + 70
        print(f'Ambient temp is {temp} \N{DEGREE SIGN} F')
        
        
ambient_temp(2)
```

```{code-cell} ipython3
time = np.array([-5,-4,-3,-2,-1,0,1,2])
ambient_temps = np.array([50,51,55,60,65,70,75,80])
m, b = np.polyfit(time, ambient_temps, deg=1)

plt.plot(time, m*time + b, '-', label = f'Line of best fit: {m:.2f}t + {b:.2f}')
plt.plot(time, ambient_temps, 'o')
plt.title('Ambient Temperature ')
plt.xlabel('Time starting at 11 am(hr)')
plt.ylabel('Temp \N{DEGREE SIGN} F')
plt.legend();
```
