import sys
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from plotly.express.colors import sample_colorscale

#local
sys.path.append('/home/parrama/Documents/Work/PhD/Scripts/Python/general/')
sys.path.append('/home/parrama/Documents/Work/PhD/Scripts/Python/observations/spectral_analysis/')
sys.path.append('/home/parrama/Documents/Work/PhD/Scripts/Python/modeling/PyXstar')
#online
sys.path.append('/app/winds/observations/spectral_analysis/')
sys.path.append('/app/winds/modeling/PyXstar')

from simul_tools import func_density_sol,func_vel_sol,func_logxi_sol
from gricad_tools import load_solutions,sample_angle,interp_yaxis

try:
    st.set_page_config(page_icon=":magnet:",layout='wide')
except:
    pass

solutions_path='/media/parrama/SSD/Simu/MHD/solutions/nathan_init/nathan_init.txt'

possible_sols_path='/media/parrama/SSD/Simu/MHD/solutions/nathan_init/super_a_0.0.dat'


sol_split=load_solutions(solutions_path,mode='file',split_sol=True,split_par=True)
sol_splitsol=load_solutions(solutions_path,mode='file',split_sol=True,split_par=False)

possible_sol=np.loadtxt(possible_sols_path,skiprows=1)

eps_values=np.array([elem[0][0][0][0] for elem in sol_split])

#selecting the epsilon values
with st.sidebar:
    eps_select=st.selectbox('Epsilon value',np.array(eps_values))

i_eps=np.argwhere(eps_values==eps_select)[0][0]

sol_indiv_eps=sol_split[i_eps]

possible_sol_indiv_eps=possible_sol[possible_sol.T[8]==eps_select]

possible_sol_indiv_p_mu=possible_sol_indiv_eps.T[:2]

with st.sidebar:
    st.header('Angle Sampling (linear)')
    split_angle=st.checkbox('Emulate angle sampling',value=False)

    mdot_obs=st.number_input(r'Observed $\dot m$',value=0.111,min_value=1e-10,format='%.3e')
    m_BH=st.number_input(r'Black Hole Mass ($M_\odot$)',value=8.,min_value=1e-10,format='%.3e')
    rj=st.number_input(r'internal WED radius ',value=6,format='%.2e')

    val_angle_low=st.number_input(r'angle interval lower limit',value=30.,format='%.2f')
    val_angle_high=st.number_input(r'angle interval upper limit',value=80.,format='%.2f')
    val_angle_step=st.number_input(r'angle step',value=4.,format='%.2f')

    st.header('SED')
    val_L_source=st.number_input(r'Source bolometric luminosity ($10^{38} erg/s/cm^2$)',value=1e0,
                                 format='%.3e')*1e38

#fetching the p/mu space of existing solutions
p_mu_space=[]

B_ratio_SM=[]
for i in range(len(sol_indiv_eps)):
    for j in range(len(sol_indiv_eps[i])):
        #taking the first line of the individual solution (doesn't matter)
        p_mu_space+=[sol_indiv_eps[i][j][0][2:4]]

        #fetching the slow magnetosonic point's z over r (aka y)
        elem_z_over_r_SM=sol_indiv_eps[i][j][0][-2]

        #and the sampling closest to this value
        line_SM=sol_indiv_eps[i][j][np.argmin(abs(sol_indiv_eps[i][j].T[7]-elem_z_over_r_SM))]

        # B_ratio_SM+=[(line_SM[15]/line_SM[16])**2]


p_mu_space=np.array([elem for elem in p_mu_space]).T



fig_scatter=go.Figure()
fig_scatter.update_layout(width=1500,height=800,)
fig_scatter.update_xaxes(type="log")
fig_scatter.update_yaxes(type="log")
fig_scatter.layout.yaxis.color = 'white'
fig_scatter.layout.xaxis.color = 'white'
fig_scatter.layout.yaxis.gridcolor='rgba(0.5,0.5,.5,0.2)'
fig_scatter.layout.xaxis.gridcolor='rgba(0.5,0.5,.5,0.2)'

fig_scatter.update_layout(xaxis=dict(showgrid=True),
              yaxis=dict(showgrid=True),xaxis_title='mu', yaxis_title='p (xi)',
                font=dict(size=16),paper_bgcolor='rgba(0.,0.,0.,0.)',plot_bgcolor='rgba(0.,0.,0.,0.)')


#plotting the scatter of the whole solution space

scat_possible=go.Scatter(x=possible_sol_indiv_p_mu[1],y=possible_sol_indiv_p_mu[0],mode='markers',
                            marker=dict(size=6,color='grey'),name='possible solutions', )

#and the one of the ones for which we have the MHD solution
scat_mhd=go.Scatter(x=p_mu_space[1],y=p_mu_space[0],mode='markers',
                            marker=dict(size=12,color='grey',line=dict(width=2,color='white')),
                    name='computed MHD solutions',hoverinfo="skip")

fig_scatter.add_trace(scat_possible)
fig_scatter.add_trace(scat_mhd)

#updating legend
fig_scatter.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99,font=dict(color='white')),hovermode='closest',    hoverlabel=dict(
        bgcolor='rgba(0.,0.,0.,0.)',font=dict(color="white")))

# fig_scatter.update_xaxes(tickfont=dict(color='white'),title_font_color="white")
# fig_scatter.update_yaxes(tickfont=dict(color='white'),title_font_color="white")

tab_p_mu, tab_sol, tab_sol_radial= st.tabs(["Solution selection", "Angular distribution", "radial distribution"])


with tab_p_mu:
    selected_points = plotly_events(fig_scatter)

with tab_sol:
    if len(selected_points)==0:
        st.info('Click on a point for which the MHD solution has been computed to display it.')
        st.stop()
    if selected_points is not None:
        selected_sol_p_mu=np.array([selected_points[0]['y'],selected_points[0]['x']])


#fetching the individual solutions with these parameters (should only be one solution like this)

selected_sol_mask=np.all(np.array([elem[0][2:4] for elem in sol_splitsol])==selected_sol_p_mu,axis=1)

selected_mhd_sol=sol_splitsol[selected_sol_mask][0]

sol_z_over_r=selected_mhd_sol.T[7]
sol_angle=selected_mhd_sol.T[8]

sol_r_cyl_r0=selected_mhd_sol.T[9]
sol_rho_mhd=selected_mhd_sol.T[10]
sol_t_mhd=selected_mhd_sol.T[14]

sol_ur,sol_uphi,sol_uz=selected_mhd_sol.T[11:14]
sol_br,sol_bphi,sol_bz=selected_mhd_sol.T[15:18]

#start of ideal MHD
sol_y_id=selected_mhd_sol.T[19][0]

#slow magnetosonic
sol_y_sm=selected_mhd_sol.T[20][0]

#Alfven
sol_y_A=selected_mhd_sol.T[21][0]


#converting these last 3 into angles
def y_to_ang(y):
    '''
    output in degrees
    '''
    return 90-(180/(np.pi)*np.arctan(y))

sol_angle_id=y_to_ang(sol_y_id)
sol_angle_sm=y_to_ang(sol_y_sm)
sol_angle_A=y_to_ang(sol_y_A)

def plotly_line_wrapper(x,y,log_x=False,log_y='auto',xaxis_title='',yaxis_title='',legend=False,
                        line_color='lightskyblue'):

    # '''
    # Wrapper to render the line in a "nice" streamlit theme while keeping the latex displays
    # '''
    # if log_y=='sim':
    #     '''
    #     arcsin scale to approximate a symlog scale
    #
    #     set up so as to leave 3 orders of magnitude below the power of 10 range of the
    #      highest point in the scale before becoming linear
    #     '''
    #
    #     y_use=np.arcsinh(y/2)/np.log(10)
    # else:
    #     y_use=y

    fig_line = go.Figure()

    neg_log=False

    if log_x:
        fig_line.update_xaxes(type="log")
    if log_y=='auto':
        #testing if there's no sign change in the y axis
        setup_log=(y[y!=0]/abs(y[y!=0])==1).all() or (y[y!=0]/abs(y[y!=0])==-1).all()

        neg_log=(y[y!=0]/abs(y[y!=0])==-1).all()
    elif log_y:
        setup_log=True

    if setup_log:
        fig_line.update_yaxes(type="log")

    line=go.Scatter(x=x,y=abs(y) if neg_log else y,line=dict(color=line_color),name='',showlegend=False)

    y_title=r'$-'+str(yaxis_title.replace('$',''))+'$' if neg_log else yaxis_title
    fig_line.update_layout(xaxis=dict(showgrid=True,zeroline=False),
                                yaxis=dict(showgrid=True,zeroline=False),
                       xaxis_title=xaxis_title, yaxis_title=y_title,
                                font=dict(size=14),
                                paper_bgcolor='rgba(0.,0.,0.,0.)', plot_bgcolor='rgba(0.,0.,0.,0.)',
                       margin=dict(l=100, r=20, t=0, b=100))


    fig_line.add_traces(line)

    fig_line.layout.xaxis.color = 'white'
    # line.layout.xaxis.zerolinecolor = 'rgba(0.5,0.5,.5,0.3)'
    fig_line.layout.xaxis.gridcolor = 'rgba(0.5,0.5,.5,0.3)'
    
    fig_line.layout.yaxis.color = 'white'
    # line.layout.yaxis.zerolinecolor = 'rgba(0.5,0.5,.5,0.3)'
    fig_line.layout.yaxis.gridcolor = 'rgba(0.5,0.5,.5,0.3)'

    if legend is False:
        fig_line.update_layout(showlegend=False)
    else:
        if legend is True:
            fig_line.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99, font=dict(color='white')), hovermode='closest', hoverlabel=dict(
                bgcolor='rgba(0.,0.,0.,0.)', font=dict(color="white")))
        elif legend == 'bot_left':
            fig_line.update_layout(legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="left",
                x=0.01, font=dict(color='white')), hovermode='closest', hoverlabel=dict(
                bgcolor='rgba(0.,0.,0.,0.)', font=dict(color="white")))

        elif legend =='top_left':
            fig_line.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01, font=dict(color='white')), hovermode='closest', hoverlabel=dict(
                bgcolor='rgba(0.,0.,0.,0.)', font=dict(color="white")))

    return fig_line

def angle_plot(x,y,log_x=False,log_y='auto',xaxis_title='',yaxis_title='',legend=False,sampl_angles=None,compt_angle=None):

    fig_line=plotly_line_wrapper(x,y,log_x=log_x,log_y='auto',xaxis_title=xaxis_title,yaxis_title=yaxis_title,legend=legend)

    if log_y=='auto':
        neg_log=(y[y!=0]/abs(y[y!=0])==-1).all()
    else:
        neg_log=False

    #adding the specific points
    #interpolating
    id_mhd_point=go.Scatter(x=[sol_angle_id],y=[(-1 if neg_log else 1)*interp_yaxis(sol_angle_id,x,y)],mode='markers',name='id mhd point',marker=dict(size=13,symbol='star-triangle-up-dot',color='black',line=dict(color='violet',width=2)))

    sm_point = go.Scatter(x=[sol_angle_sm], y=[(-1 if neg_log else 1)*interp_yaxis(sol_angle_sm, x, y)], mode='markers', name='sm point',marker=dict(size=13,symbol='star-triangle-down-dot',color='black',line=dict(color='green',width=2)))

    Alfven_point = go.Scatter(x=[sol_angle_A],y=[(-1 if neg_log else 1)*interp_yaxis(sol_angle_A, x, y)], mode='markers', name='Alfven point',marker=dict(size=13,symbol='circle-x',color='black',line=dict(color='red',width=2)))

    fig_line.add_traces(id_mhd_point)
    fig_line.add_traces(sm_point)
    fig_line.add_traces(Alfven_point)

    if sampl_angles is not None:
        mask_sampl_angles=np.array([sol_angle==elem for elem in sampl_angles]).any(0)

        sampl_points=go.Scatter(x=x[mask_sampl_angles],y=(-1 if neg_log else 1)*y[mask_sampl_angles], mode='markers', name='angle sampling',marker=dict(size=10,symbol='line-ns',line=dict(color='orange',width=2)))
        fig_line.add_traces(sampl_points)

    if compt_angle is not None:
        fig_line.add_vrect(x0=compt_angle, x1=90, line_width=0, fillcolor="grey", opacity=0.2,
                           annotation_text="compton-thick", annotation_position="bottom",
                           annotation=dict(font=dict(color='white')))

    st.components.v1.html(fig_line.to_html(include_mathjax='cdn'), width=500, height=450)

if split_angle:

    array_sampl_angle,compton_angles_arr=sample_angle(solutions_path,
                                          angle_values=np.arange(val_angle_low,val_angle_high+0.00001,val_angle_step),
                                          mdot_obs=mdot_obs,m_BH=m_BH,r_j=rj,mode='array',return_compton_angle=True)

    sol_split_angle=load_solutions(array_sampl_angle,mode='array',split_sol=True,split_par=False)

    selected_sol_split_angle=sol_split_angle[selected_sol_mask][0]

    selected_angles=selected_sol_split_angle.T[8]

    compton_angle=compton_angles_arr[selected_sol_mask][0]

else:
    selected_angles=None
    compton_angle=None

with tab_sol:

    st.title(r'solution: $\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\textrm{p}='+str(round(selected_sol_p_mu[0],4))+
             '\;\;\;|\;\;\; \mu='+str(round(selected_sol_p_mu[1],4))+'$')
    #now we can display the plots of the individual solution
    col_1,col_2,col_3=st.columns(3)


    with col_1:

        angle_plot(sol_angle,sol_r_cyl_r0,log_x=False,log_y=True,
                            xaxis_title=r'$\theta \; (°)$',yaxis_title=r'$r_{cyl}/r_0$',legend=True,
                            sampl_angles=selected_angles,compt_angle=compton_angle)

        angle_plot(sol_angle,sol_ur,log_x=False,
                            xaxis_title=r'$\theta \; (°)$', yaxis_title=r'$u_{r}$',legend=True,
                            sampl_angles=selected_angles,compt_angle=compton_angle)

        angle_plot(sol_angle,y=sol_br,log_x=False,
                            xaxis_title=r'$\theta \; (°)$', yaxis_title=r'$B_r$',legend='top_left',
                            sampl_angles=selected_angles,compt_angle=compton_angle)

    with col_2:

        angle_plot(x=sol_angle,y=sol_rho_mhd,log_x=False,log_y=True,
                            xaxis_title =r'$\theta \; (°)$', yaxis_title = r'$\rho_{mhd}$',
                            sampl_angles=selected_angles,compt_angle=compton_angle)

        angle_plot(x=sol_angle, y=sol_uphi,log_x=False,log_y=True,
                            xaxis_title=r'$\theta \; (°)$', yaxis_title=r'$u_{\phi}$',
                            sampl_angles=selected_angles,compt_angle=compton_angle)

        angle_plot(x=sol_angle, y=sol_bphi,log_x=False,
                            xaxis_title=r'$\theta \; (°)$', yaxis_title=r'$B_{\phi}$',
                            sampl_angles=selected_angles,compt_angle=compton_angle)


    with col_3:

        angle_plot(x=sol_angle,y=sol_t_mhd,log_x=False,log_y=True,
                            xaxis_title=r'$\theta \; (°)$', yaxis_title=r'$T_{mhd}$',
                            sampl_angles=selected_angles,compt_angle=compton_angle)

        angle_plot(x=sol_angle, y=sol_uz,log_x=False,
                            xaxis_title=r'$\theta \; (°)$', yaxis_title=r'$u_{z}$',
                            sampl_angles=selected_angles,compt_angle=compton_angle)

        angle_plot(x=sol_angle, y=sol_bz,log_x=False,
                            xaxis_title=r'$\theta \; (°)$', yaxis_title=r'$B_z$',
                            sampl_angles=selected_angles,compt_angle=compton_angle)

if not split_angle:
    sys.exit()

def radial_plot(rad,sol_sampl,angl_sampl,log_x=False,log_y=False,xaxis_title='',yaxis_title='',legend=False,
                cmap='plasma_r'):

    norm_angl= (angl_sampl-val_angle_low)/(val_angle_high-val_angle_low)

    ang_colors = sample_colorscale(cmap,norm_angl)

    #creating the theme with the first line
    fig_rad=plotly_line_wrapper(rad,sol_sampl[0],log_x=log_x,log_y='auto',xaxis_title=xaxis_title,yaxis_title=yaxis_title,legend=False,line_color=ang_colors[0])

    #and adding the rest of the lines
    for elem_sampl,elem_angl,elem_color in zip(sol_sampl[1:],angl_sampl[1:],ang_colors[1:]):
        fig_rad.add_trace(
        go.Scatter(x=rad,y=elem_sampl,line=dict(color=elem_color),name='',showlegend=False))

    #tickvals for the cmap with some rounding
    tickvals_cm=np.array([round(elem,1) for elem in angl_sampl.tolist()+[val_angle_low,val_angle_high]])

    colorbar_trace = go.Scatter(x=[None],y=[None],mode='markers',
                                marker=dict(colorscale=cmap,showscale=True,
                                            cmin=val_angle_low,cmax=val_angle_high,
                                    colorbar=dict(thickness=10, tickvals=tickvals_cm,tickfont=dict(color='white'),
                                                  ticks='outside',ticklen=3,tickcolor='white',
                                                  title='theta',titlefont=dict(color='white'))),hoverinfo='none')
    fig_rad.add_trace(colorbar_trace)

    fig_rad.update_layout(xaxis=dict(showgrid=True,zeroline=False),
                                yaxis=dict(showgrid=True,zeroline=False),
                       xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                                font=dict(size=14),
                                paper_bgcolor='rgba(0.,0.,0.,0.)', plot_bgcolor='rgba(0.,0.,0.,0.)',
                       margin=dict(l=100, r=20, t=0, b=100))

    st.components.v1.html(fig_rad.to_html(include_mathjax='cdn'), width=500, height=450)

with tab_sol_radial:

    sol_sampl_z_over_r = selected_sol_split_angle.T[7]
    sol_sampl_angle = selected_sol_split_angle.T[8]

    sol_sampl_r_cyl_r0 = selected_sol_split_angle.T[9]
    sol_sampl_rho_mhd = selected_sol_split_angle.T[10]

    sol_sampl_ur, sol_sampl_uphi, sol_sampl_uz = selected_sol_split_angle.T[11:14]
    sol_sampl_br, sol_sampl_bphi, sol_sampl_bz = selected_sol_split_angle.T[15:18]

    sol_p_mhd=selected_sol_split_angle[0][3]

    mdot_mhd=mdot_obs*12

    r_sph_sampling=np.logspace(np.log10(rj),6,1000)

    rho_sampl=np.array([func_density_sol(r_sph_sampling,sol_sampl_z_over_r[i],sol_sampl_rho_mhd[i],sol_p_mhd,mdot_mhd,m_BH) for\
                        i in range(len(sol_sampl_z_over_r))])

    logxi_sampl=np.array([func_logxi_sol(r_sph_sampling,sol_sampl_z_over_r[i],val_L_source,sol_sampl_rho_mhd[i],sol_p_mhd,mdot_mhd,m_BH) for\
                       i in range(len(sol_sampl_z_over_r))])

    #here all speeds are divided by 1e5 to go to km/s
    ur_sampl=np.array([func_vel_sol('r',r_sph_sampling,sol_sampl_z_over_r[i],sol_sampl_ur[i],sol_sampl_uphi[i],sol_sampl_uz[i]) for\
                       i in range(len(sol_sampl_z_over_r))])/1e5

    uphi_sampl=np.array([func_vel_sol('phi',r_sph_sampling,sol_sampl_z_over_r[i],sol_sampl_ur[i],sol_sampl_uphi[i],sol_sampl_uz[i]) for\
                       i in range(len(sol_sampl_z_over_r))])/1e5

    uz_sampl=np.array([func_vel_sol('z',r_sph_sampling,sol_sampl_z_over_r[i],sol_sampl_ur[i],sol_sampl_uphi[i],sol_sampl_uz[i]) for\
                       i in range(len(sol_sampl_z_over_r))])/1e5

    uobs_sampl=np.array([func_vel_sol('obs',r_sph_sampling,sol_sampl_z_over_r[i],sol_sampl_ur[i],sol_sampl_uphi[i],sol_sampl_uz[i]) for\
                       i in range(len(sol_sampl_z_over_r))])/1e5

    col_a,col_b,col_c=st.columns(3)

    with col_a:
        radial_plot(r_sph_sampling,rho_sampl,sol_sampl_angle,log_x=True,
                                    log_y=True,xaxis_title=r'$R_{sph}\;$ (Rg)',yaxis_title=r'$n$ (cgs)')

        radial_plot(r_sph_sampling,ur_sampl,sol_sampl_angle,log_x=True,
                                    log_y=True,xaxis_title=r'$R_{sph}\;$ (Rg)',yaxis_title=r'$v_{r}$ (km/s)')

    with col_b:
        radial_plot(r_sph_sampling,logxi_sampl, sol_sampl_angle, log_x=True,
                    log_y=False, xaxis_title=r'$R_{sph}\;$ (Rg)', yaxis_title=r'$log\xi$')

        radial_plot(r_sph_sampling, uphi_sampl, sol_sampl_angle, log_x=True,
                    log_y=True, xaxis_title=r'$R_{sph}\;$ (Rg)', yaxis_title=r'$v_{\phi}$ (km/s)')

    with col_c:
        radial_plot(r_sph_sampling, uobs_sampl, sol_sampl_angle, log_x=True,
                    log_y=True, xaxis_title=r'$R_{sph}\;$ (Rg)', yaxis_title=r'$v_{obs}$ (km/s)')

        radial_plot(r_sph_sampling, uz_sampl, sol_sampl_angle, log_x=True,
                    log_y=True, xaxis_title=r'$R_{sph}\;$ (Rg)', yaxis_title=r'$v_{z}$ (km/s)')