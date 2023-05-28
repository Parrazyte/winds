import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

from gricad_tools import load_solutions,sample_angle

try:
    st.set_page_config(page_icon=":hole:",layout='wide')
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
    split_angle=st.checkbox('Emulate angle sampling',value=True)

    mdot_obs=st.number_input(r'Observed $\dot m$',value=0.111,min_value=1e-10,format='%.3e')
    m_BH=st.number_input(r'Black Hole Mass ($M_\odot$)',value=8.,min_value=1e-10,format='%.3e')
    rj=st.number_input(r'internal WED radius ',value=6,format='%.2e')

    val_angle_low=st.number_input(r'angle interval lower limit',value=30.,format='%.2f')
    val_angle_high=st.number_input(r'angle interval upper limit',value=80.,format='%.2f')
    val_angle_step=st.number_input(r'angle step',value=4.,format='%.2f')


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

tab_p_mu, tab_sol, tab_sol_angle= st.tabs(["Solution selection", "Angular distribution", "radial distribution"])

with tab_p_mu:
    selected_points = plotly_events(fig_scatter)

with tab_sol:
    if len(selected_points)==0:
        st.info('Click on a point for which the MHD solution has been computed to display it.')
        st.stop()
    if selected_points is not None:
        selected_sol_p_mu=np.array([selected_points[0]['y'],selected_points[0]['x']])

#fetching the individual solutions with these parameters (should only be one solution like this)
selected_mhd_sol=sol_splitsol[np.all(np.array([elem[0][2:4] for elem in sol_splitsol])==selected_sol_p_mu,axis=1)][0]

sol_z_over_r=selected_mhd_sol.T[7]
sol_angle=selected_mhd_sol.T[8]

sol_r_cyl_r0=selected_mhd_sol.T[9]
sol_rho_mhd=selected_mhd_sol.T[10]
sol_t_mhd=selected_mhd_sol.T[14]

sol_ur,sol_uphi,sol_uz=selected_mhd_sol.T[11:14]
sol_br,sol_bphi,sol_bz=selected_mhd_sol.T[15:18]

with tab_sol:
    #now we can display the plots of the individual solution
    col_1,col_2,col_3=st.columns(3)


    with col_1:

        line_r_cyl_r0=px.line(x=sol_z_over_r,y=sol_r_cyl_r0,log_x=True)
        line_r_cyl_r0.update_layout(xaxis=dict(showgrid=True),
                      yaxis=dict(showgrid=True),xaxis_title=r'$z/r_0$', yaxis_title=r'$r_{cyl}/r_0$',
                        font=dict(size=16),paper_bgcolor='rgba(0.,0.,0.,0.)',plot_bgcolor='rgba(0.,0.,0.,0.)')
        st.plotly_chart(line_r_cyl_r0,use_container_width=True)

        # line_rcyl_ro=px.line(x=sol_z_over_r,y=sol_r_cyl_r0,log_x=True)
        # line_rcyl_ro.update_layout(xaxis=dict(showgrid=True),
        #               yaxis=dict(showgrid=True),xaxis_title=r'$z/r_0$', yaxis_title=r'$r_{cyl}/r_0$',
        #                 font=dict(size=12),paper_bgcolor='rgba(0.,0.,0.,0.)',plot_bgcolor='rgba(0.,0.,0.,0.)')
        # line_rcyl_ro.layout.yaxis.color = 'white'
        # line_rcyl_ro.layout.xaxis.color = 'white'
        # line_rcyl_ro.layout.yaxis.gridcolor = 'rgba(0.5,0.5,.5,0.2)'
        # line_rcyl_ro.layout.xaxis.gridcolor = 'rgba(0.5,0.5,.5,0.2)'
        #
        # st.components.v1.html(line_rcyl_ro.to_html(include_mathjax='cdn'),width=600,height=600)
        # # st.plotly_chart(line_rcyl_ro,use_container_width=True)

        line_ur=px.line(x=sol_z_over_r,y=sol_ur,log_x=True)
        line_ur.update_layout(xaxis=dict(showgrid=True),
                      yaxis=dict(showgrid=True),xaxis_title=r'$z/r_0$', yaxis_title=r'$u_{r}$',
                        font=dict(size=16),paper_bgcolor='rgba(0.,0.,0.,0.)',plot_bgcolor='rgba(0.,0.,0.,0.)')
        st.plotly_chart(line_ur,use_container_width=True)

        line_br=px.line(x=sol_z_over_r,y=sol_br,log_x=True)
        line_br.update_layout(xaxis=dict(showgrid=True),
                      yaxis=dict(showgrid=True),xaxis_title=r'$z/r_0$', yaxis_title=r'$B_r$',
                        font=dict(size=16),paper_bgcolor='rgba(0.,0.,0.,0.)',plot_bgcolor='rgba(0.,0.,0.,0.)')
        st.plotly_chart(line_br,use_container_width=True)

    with col_2:
        line_rho_mhd=px.line(x=sol_z_over_r,y=sol_rho_mhd,log_x=True)
        line_rho_mhd.update_layout(xaxis=dict(showgrid=True),
                      yaxis=dict(showgrid=True),xaxis_title=r'$z/r_0$', yaxis_title=r'$\rho_{mhd}$',
                        font=dict(size=16),paper_bgcolor='rgba(0.,0.,0.,0.)',plot_bgcolor='rgba(0.,0.,0.,0.)')
        st.plotly_chart(line_rho_mhd,use_container_width=True)

        line_uphi = px.line(x=sol_z_over_r, y=sol_uphi,log_x=True)
        line_uphi.update_layout(xaxis=dict(showgrid=True),
                              yaxis=dict(showgrid=True), xaxis_title=r'$z/r_0$', yaxis_title=r'$u_{\phi}$',
                              font=dict(size=16), paper_bgcolor='rgba(0.,0.,0.,0.)', plot_bgcolor='rgba(0.,0.,0.,0.)')
        st.plotly_chart(line_uphi,use_container_width=True)

        line_bphi = px.line(x=sol_z_over_r, y=sol_bphi,log_x=True)
        line_bphi.update_layout(xaxis=dict(showgrid=True),
                              yaxis=dict(showgrid=True), xaxis_title=r'$z/r_0$', yaxis_title=r'$B_{\phi}$',
                              font=dict(size=16), paper_bgcolor='rgba(0.,0.,0.,0.)', plot_bgcolor='rgba(0.,0.,0.,0.)')
        st.plotly_chart(line_bphi,use_container_width=True)

    with col_3:
        line_t_mhd=px.line(x=sol_z_over_r,y=sol_t_mhd,log_x=True)
        line_t_mhd.update_layout(xaxis=dict(showgrid=True),
                      yaxis=dict(showgrid=True),xaxis_title=r'$z/r_0$', yaxis_title=r'$T_{mhd}$',
                        font=dict(size=16),paper_bgcolor='rgba(0.,0.,0.,0.)',plot_bgcolor='rgba(0.,0.,0.,0.)')
        st.plotly_chart(line_t_mhd,use_container_width=True)

        line_uz = px.line(x=sol_z_over_r, y=sol_uz,log_x=True)
        line_uz.update_layout(xaxis=dict(showgrid=True),
                              yaxis=dict(showgrid=True), xaxis_title=r'$z/r_0$', yaxis_title=r'$u_{z}$',
                              font=dict(size=16), paper_bgcolor='rgba(0.,0.,0.,0.)', plot_bgcolor='rgba(0.,0.,0.,0.)')
        st.plotly_chart(line_uz,use_container_width=True)

        line_bz = px.line(x=sol_z_over_r, y=sol_bz,log_x=True)
        line_bz.update_layout(xaxis=dict(showgrid=True),
                              yaxis=dict(showgrid=True), xaxis_title=r'$z/r_0$', yaxis_title=r'$B_z$',
                              font=dict(size=16), paper_bgcolor='rgba(0.,0.,0.,0.)', plot_bgcolor='rgba(0.,0.,0.,0.)')
        st.plotly_chart(line_bz,use_container_width=True)

#st.plotly_chart(fig_scatter,theme=None)

with tab_sol_angle:
    if split_angle:
        sol_split_angle=load_solutions(sample_angle(solutions_path,
                                              angle_values=np.arange(val_angle_low,val_angle_high+0.00001,val_angle_step),
                                              mdot_obs=mdot_obs,m_BH=m_BH,r_j=rj,mode='array'),mode='array',
                                              split_sol=True,split_par=True)
