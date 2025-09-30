#load

os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1744-294/new_anal/fit_duo')
Xset.restore('mod_NSdg_contonly.xcm')

rebinv_xrism(1,5)

fig = plt.figure(figsize=(8.75, 7))
grid = GridSpec(len('ldata,delchi'.split(',')), 1, figure=fig, hspace=0.)
axes = [plt.subplot(elem) for elem in grid]

xPlot('ldata,delchi',axes_input=axes)
fig=plt.gca().get_figure()
fig.get_children()[1].get_children()[4].remove()
fig.get_children()[1].get_children()[4].remove()
fig.get_children()[1].get_children()[-2].remove()
fig.get_children()[2].get_children()[-2].remove()
plot_std_ener(fig.get_children()[2],plot_indiv_transi=True,mode='',alpha_line=0.3,noname=True)
plot_std_ener(fig.get_children()[1],plot_indiv_transi=True,mode='',alpha_line=0.3,noname=True)
plot_std_ener(fig.get_children()[2],plot_indiv_transi=False,mode='misc',alpha_line=0.5,noline=True)


axes[1].tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=True,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=True,
    labeltop=False,
    direction='out')


plt.subplots_adjust(left=0.08)
plt.subplots_adjust(right=0.98)
plt.subplots_adjust(bottom=0.07)
plt.subplots_adjust(top=0.93)