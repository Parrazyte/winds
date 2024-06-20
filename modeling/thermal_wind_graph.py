import matplotlib.pyplot as plt
import numpy as np

# Create figure and axis
fig, ax = plt.subplots()

# Set the scale to logarithmic
ax.set_xscale('log')
ax.set_yscale('log')

# Define the limits
ax.set_xlim(0.01, 100)
ax.set_ylim(0.001, 100)

# Plot the lines
ax.plot([0.01, 1], [1, 1], 'k-', lw=1)  # Horizontal line at y=1
ax.plot([1, 1], [0.001, 100], 'k-', lw=1)  # Vertical line at x=1

'''
Lcrit = 0.03*(T_IC,8**(1/2)) LEdd from begelman83 (see vittoria's ufo paper)
'''
ax.plot([1, 100], [1, 100], 'k-', lw=1)  # Diagonal line in region A

ax.plot([1, 20], [1, 1e-3], 'k-', lw=1)  # Dashed diagonal line in region B
ax.plot([1, 100], [1, 1e-2],color='black',ls='--', lw=1)  # Solid diagonal line in region C

# Add region labels
ax.text(0.25, 10, 'E', fontsize=12, va='center')
ax.text(3, 10, 'A', fontsize=12, va='center')
ax.text(3, 1, 'B', fontsize=12, va='center')
ax.text(3, 0.01, 'C', fontsize=12, va='center')
ax.text(0.25, 0.01, 'D', fontsize=12, va='center')

# Set labels
ax.set_xlabel(r'$R/R_{IC}$')
ax.set_ylabel(r'$L/L_{crit}$')

t_cold=2e7
t_warm=2e8

def r_ic_cold(rg):
    return 6.4e4*1e8/t_cold*rg

def rg_cold(r_ic):
    return r_ic/(6.4e4*1e8/t_cold)
def r_ic_warm(rg):
    return 6.4e4*1e8/t_warm*rg

def rg_warm(r_ic):
    return r_ic/(6.4e4*1e8/t_warm)

def l_edd_cold(l_crit):
    return (0.03*(t_cold/1e8)**(-1/2))**(-1)*l_crit

def l_crit_cold(l_edd):
    return l_edd*(0.03*(t_cold/1e8)**(-1/2))

def l_edd_warm(l_crit):
    return (0.03*(t_warm/1e8)**(-1/2))**(-1)*l_crit

def l_crit_warm(l_edd):
    return l_edd*(0.03*(t_warm/1e8)**(-1/2))

# compton radius to Rg for low Compton Temperature
rg_cold_ax = ax.secondary_xaxis(1., functions=(r_ic_cold,rg_cold))

rg_cold_ax.set_xlabel(r'$R/R_{g}$ in soft state')

# compton radius to Rg for low Compton Temperature
rg_warm_ax = ax.secondary_xaxis(1.3, functions=(r_ic_warm,rg_warm))

rg_warm_ax.set_xlabel(r'$R/R_{g}$ in hard state')



# compton radius to Rg for low Compton Temperature
l_edd_cold_ax = ax.secondary_yaxis(1., functions=(l_crit_cold,l_edd_cold))

l_edd_cold_ax.set_ylabel(r'$L/L_{Edd}$ in soft state')

l_edd_warm_ax = ax.secondary_yaxis(1.3, functions=(l_crit_warm,l_edd_warm))

l_edd_warm_ax.set_ylabel(r'$L/L_{Edd}$ in hard state')

plt.subplots_adjust(right=0.7,top=0.7)
# Set grid
# ax.grid(True, which="both", ls="-")

plt.show()

# Create figure and axis
fig, ax = plt.subplots(figsize=(6,6))

plt.subplots_adjust(right=0.8,top=0.8)

# Set the scale to logarithmic
ax.set_xscale('log')
ax.set_yscale('log')

# Define the limits
ax.set_xlim(0.01, 100)
ax.set_ylim(0.01, 100)

# Plot the lines
ax.plot([1e-2, 0.25],[2, 2], 'k-', lw=1)  # Horizontal line at y=1
ax.plot([0.25, 0.25], [0.001, 100], 'k-', lw=1)  # Vertical line at x=1

'''
Lcrit = 0.03*(T_IC,8**(1/2)) LEdd from begelman83 (see vittoria's ufo paper)
'''
ax.plot([0.25, 250], [2, 2000], 'k-', lw=1)  # Diagonal line in region A

# Diagonal line B-C high (slope of -1, see page 791)
ax.plot([0.25, 2], [2, 0.25], 'k-', lw=1)

# Diagonal line B-C low (slope of -2, see page 791 giving R(L) instead of L(R))
ax.plot([2,100],[0.25,0.25*(50)**(-2)], 'k-', lw=1)

# Add region labels
ax.text(0.05, 13, 'isothermal\n corona\nE', fontsize=12, va='bottom',ha='center')
ax.text(0.9, 13, 'isothermal\n wind\nA', fontsize=12, va='bottom',ha='center')
ax.text(10, 2, 'B\nsteadily heated wind', fontsize=12, va='center',ha='center')
ax.text(1.2, 0.08, 'C\n Gravity\ninhibited Wind', fontsize=12, va='top',ha='center')
ax.text(0.05, 0.08, 'D\nNon-isothermal\n corona', fontsize=12, va='top',ha='center')

# Set labels
ax.set_xlabel(r'$R/R_{IC}$')
ax.set_ylabel(r'$L/L_{crit}$')

t_cold=1.3e7
t_warm=2e8

def r_ic_cold(rg):
    return 6.4e4*1e8/t_cold*rg

def rg_cold(r_ic):
    return r_ic/(6.4e4*1e8/t_cold)

def l_edd_cold(l_crit):
    return (0.03*(t_cold/1e8)**(-1/2))**(-1)*l_crit

def l_crit_cold(l_edd):
    return l_edd*(0.03*(t_cold/1e8)**(-1/2))



# compton radius to Rg for low Compton Temperature
rg_cold_ax = ax.secondary_xaxis(1., functions=(r_ic_cold,rg_cold))

rg_cold_ax.set_xlabel(r'$R/R_{g}$ for $T_{IC}=1.3\cdot10^{7}$K')


# compton radius to Rg for low Compton Temperature
l_edd_cold_ax = ax.secondary_yaxis(1., functions=(l_crit_cold,l_edd_cold))

l_edd_cold_ax.set_ylabel(r'$L/L_{Edd}$ for $T_{IC}=1.3\cdot10^{7}$K')


#just set up to the bounds
# wind_line=ax.fill_between([0.25,2],[2, 0.25],[100, 100],color='green',alpha=0.5)
# wind_line=ax.fill_between([2,100],[0.25,0.25*(50)**(-2)],[100, 100],color='green',alpha=0.5)

wind_line=ax.fill_between([0.25,2,100],[2, 0.25,0.25*(50)**(-2)],[100, 100,100],color='green',alpha=0.5)

nowind_line=ax.fill_between([1e-2,0.25],[1e-2,1e-2],[100, 100],color='red',alpha=0.5)
nowind_line2=ax.fill_between([0.25,2,100],[1e-2,1e-2,1e-2],[2, 0.25,0.25*(50)**(-2)],color='red',alpha=0.5)


# Create figure and axis
fig, ax = plt.subplots(figsize=(6,6))

plt.subplots_adjust(right=0.8,top=0.8)

# Set the scale to logarithmic
ax.set_xscale('log')
ax.set_yscale('log')

# Define the limits
ax.set_xlim(0.01, 100)
ax.set_ylim(0.01, 100)
# ax.set_xlim(10**-1.5,10**1.5)
# ax.set_ylim(10**-1.5,10**1.5)

# Plot the lines
ax.plot([1e-2, 0.25],[2, 2], 'k-', lw=1)  # Horizontal line at y=1
ax.plot([0.25, 0.25], [0.001, 100], 'k-', lw=1,zorder=10)  # Vertical line at x=1

'''
Lcrit = 0.03*(T_IC,8**(1/2)) LEdd from begelman83 (see vittoria's ufo paper)
'''
ax.plot([0.25, 100], [2, 800], 'k-', lw=1)  # Diagonal line in region A

#slope accoridng to page 791, once again to invert
ax.plot([0.25, 100], [2, 2*(100/0.25)**(-3/4)], 'k-', lw=1)  # Diagonal line B-C

# ax.plot([1.89,5.22],[0.25,10**(-1.5)], 'k-', lw=1) # Diagonal line B-C low

# Add region labels
ax.text(0.05, 13, 'isothermal\n corona\nE', fontsize=12, va='bottom',ha='center')
ax.text(0.9, 13, 'isothermal\n wind\nA', fontsize=12, va='bottom',ha='center')
ax.text(10, 2, 'B\nsteadily heated wind', fontsize=12, va='center',ha='center')
ax.text(2.5, 0.08, 'C\n Gravity\ninhibited Wind', fontsize=12, va='top',ha='center')
ax.text(0.05, 0.08, 'D\nNon-isothermal\n corona', fontsize=12, va='top',ha='center')

# Set labels
ax.set_xlabel(r'$R/R_{IC}$')
ax.set_ylabel(r'$L/L_{crit}$')

t_warm=1e8

def r_ic_warm(rg):
    return 6.4e4*1e8/t_warm*rg

def rg_warm(r_ic):
    return r_ic/(6.4e4*1e8/t_warm)

def l_edd_warm(l_crit):
    return (0.03*(t_warm/1e8)**(-1/2))**(-1)*l_crit

def l_crit_warm(l_edd):
    return l_edd*(0.03*(t_warm/1e8)**(-1/2))



# compton radius to Rg for low Compton Temperature
rg_warm_ax = ax.secondary_xaxis(1., functions=(r_ic_warm,rg_warm))

rg_warm_ax.set_xlabel(r'$R/R_{g}$ for $T_{IC}=10^{8}$K')

#just set up to the bounds
wind_line=ax.fill_between([0.25,100],[2, 2*(100/0.25)**(-3/4)],[100, 100],color='green',alpha=0.5)

nowind_line=ax.fill_between([1e-2,0.25],[1e-2,1e-2],[100, 100],color='red',alpha=0.5)
nowind_line2=ax.fill_between([0.25,100],[1e-2,1e-2],[2, 2*(100/0.25)**(-3/4)],color='red',alpha=0.5)


# compton radius to Rg for low Compton Temperature
l_edd_warm_ax = ax.secondary_yaxis(1., functions=(l_crit_warm,l_edd_warm))

l_edd_warm_ax.set_ylabel(r'$L/L_{Edd}$ for $T_{IC}=10^{8}$K')


