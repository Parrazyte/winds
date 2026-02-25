

'''
The goal here was to test if the radial evolution of the disk temperature (R**-3/4) could impact the
ionization parameter in the outer disk and for instance turn a stable wind unstable in the soft state
(where we would switch from an ionizing flux only close to the center to a diskbb with radial evolution
For this we compared the flux at infinity to the flux at a closer value close to the disk
C is N times R_out to make it simpler and avoid continuity issues

it seems that in total luminosity the difference is negligible
question is: does the total luminosity matter or is it mostly electrons at lower energy (aka further)
Then it could change something depending on where these electrons are


'''
def prim_flux(R,C):

    #from wolfram alpha
    val=(C*(1/(C-R)-1/R)+np.log((R/(R-C))**2))/C**3

    return val

range_Rout_log=[2,3,4,5]

for i,Rout_val in enumerate(range_Rout_log):

    C_vals=np.logspace(Rout_val+0.5,Rout_val+2.5)

    f_vals=[(prim_flux(10**Rout_val,elem_C_val)-prim_flux(1,elem_C_val))/(1/elem_C_val**2) for elem_C_val in C_vals]

    plt.figure()
    plt.xlabel('R_wind/R_out')
    plt.ylabel('flux/flux_at_inf')
    plt.xscale('log')
    plt.yscale('log')
    plt.suptitle('R_out='+str(10**Rout_val))
    plt.plot(C_vals/10**Rout_val,f_vals)