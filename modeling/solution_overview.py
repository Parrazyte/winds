import os,sys
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from plotly.express.colors import sample_colorscale
from plotly.subplots import make_subplots

import scipy

from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

#local
sys.path.append('/home/parrama/Documents/Work/PhD/Scripts/Python/general/')
sys.path.append('/home/parrama/Documents/Work/PhD/Scripts/Python/observations/spectral_analysis/')
sys.path.append('/home/parrama/Documents/Work/PhD/Scripts/Python/modeling/PyXstar')
#online
sys.path.append('/app/winds/observations/spectral_analysis/')
sys.path.append('/app/winds/modeling/PyXstar')

from general_tools import interval_extract,ravel_ragged

from solution_tools import func_density_sol,func_vel_sol,func_logxi_sol,func_nh_sol,load_solutions,sample_angle,interp_yaxis

sigma_thomson_cgs = 6.6524587321e-25
c_cgs = 2.99792458e10
G_cgs = 6.6743015e-8
Msol_cgs = 1.98847e33

k_boltzmann_cgs=1.380649e-16
sigma_boltzmann_cgs=5.670374e-5
m_p_cgs=1.67262192e-24



try:
    st.set_page_config(page_icon=":magnet:",layout='wide')
except:
    pass

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

#rough way of testing if online or not
online='parrama' not in os.getcwd()

if not online:
    update_online = st.sidebar.button('Update online version')

    solutions_path='/media/parrama/SSD/Simu/MHD/solutions/nathan_init/nathan_init.txt'

    possible_sol_path='/media/parrama/SSD/Simu/MHD/solutions/nathan_init/super_a_0.0.dat'

    #solutions_path='/media/parrama/SSD/Simu/MHD/solutions/nathan_init/nathan_init.txt'
    #refaire un merge des trois epsilon

    #possible_sol_path='/media/parrama/SSD/Simu/MHD/solutions/nathan_new/a_alphat_0.0_ep_0.10.dat'
    #et le reste

else:

    update_online = False

    if not check_password():
        st.stop()

    solutions_path='/app/winds/modeling/solutions_dumps/nathan_init.txt'
    possible_sol_path = '/app/winds/modeling/solutions_dumps/super_a_0.0.dat'

if update_online:
    # updating script
    path_online = __file__.replace('.py', '_online.py')

    os.system('cp ' + __file__ + ' ' + path_online)

    file_dir=__file__[:__file__.rfind('/')]

    os.system('mkdir -p '+file_dir+'/solutions_dumps/')

    os.system('cp '+solutions_path+' '+file_dir+'/solutions_dumps/'+solutions_path.split('/')[-1])
    os.system('cp '+possible_sol_path+' '+file_dir+'/solutions_dumps/'+possible_sol_path.split('/')[-1])

Msol_SI = 1.98892e30
m2cm=100
c_SI = 2.99792e8
G_SI = 6.674e-11


sol_split=load_solutions(solutions_path,mode='file',split_sol=True,split_par=True)
sol_splitsol=load_solutions(solutions_path,mode='file',split_sol=True,split_par=False)

possible_sol=np.loadtxt(possible_sol_path,skiprows=1)

eps_values=np.array([elem[0][0][0][0] for elem in sol_split])

#selecting the epsilon values
with st.sidebar:
    eps_select=st.selectbox('Epsilon value',np.array(eps_values))

    sol_order_select=st.radio('selected solutions ordering',['Increasing mu','Increasing p'])

i_eps=np.argwhere(eps_values==eps_select)[0][0]

sol_indiv_eps=sol_split[i_eps]

possible_sol_indiv_eps=possible_sol[possible_sol.T[8]==eps_select]

possible_sol_indiv_p_mu=possible_sol_indiv_eps.T[:2]

#avoiding repetitions (some solutions are repeated due to being found  with different search intervals)
pos_unique=[]
for elem in possible_sol_indiv_p_mu.T:
    if len(pos_unique)==0 or not np.any(np.all(pos_unique==elem,1)):
        pos_unique+=[elem]

possible_sol_indiv_p_mu=np.array(pos_unique).T

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

    with st.expander('visualisation'):
        latex_title=st.checkbox('Force latex whenever possible(unstable)')
        visu_2D=st.checkbox('Plot 2D solution visualization (in progress)')

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
fig_scatter.update_layout(width=1500,height=750)
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

non_computed_p_mu=np.array([elem for elem in possible_sol_indiv_p_mu.T if not np.any(np.all(p_mu_space.T==elem,1))]).T

#mask_non_computed=possible_sol_indiv_p_mu[0][elem not in p_mu_space[0] fro elem in possible_sol_indiv_p_mu[0]]
scat_possible=go.Scatter(x=non_computed_p_mu[1],y=non_computed_p_mu[0],mode='markers',
                            marker=dict(size=6,color='grey'),name='possible solutions', )

#and the one of the ones for which we have the MHD solution
scat_mhd=go.Scatter(x=p_mu_space[1],y=p_mu_space[0],mode='markers',
                            marker=dict(size=12,color='grey',line=dict(width=2,color='white')),
                    name='computed MHD solutions')

fig_scatter.add_trace(scat_possible)
fig_scatter.add_trace(scat_mhd)

#updating legend
fig_scatter.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99,font=dict(color='white')),hovermode='closest',    hoverlabel=dict(
        bgcolor='rgba(0.,0.,0.,0.)',font=dict(color="white")),margin=dict(t=20))

# fig_scatter.update_xaxes(tickfont=dict(color='white'),title_font_color="white")
# fig_scatter.update_yaxes(tickfont=dict(color='white'),title_font_color="white")

tab_p_mu, tab_sol, tab_sol_radial,tab_explo,tab_2D,tab_tstruct= st.tabs(["Solution selection", "Angular distributions", "radial distribution","parameter exploration","full solution vizualisation","thermal structure"])

with tab_p_mu:
    selected_points = plotly_events(fig_scatter,click_event=True,select_event=True,override_height=800)
    st.info('Rerun if this figure disappears.')

if len(selected_points)==0:
    with tab_sol:
        st.info('Click or select point(s) for which the MHD solution has been computed to display the angular'+
                ' distribution of its/their properties.')
    with tab_sol_radial:
        st.info('Click on a point for which the MHD solution has been computed to display radial distributions.')
    with tab_explo:
        st.info('Click on a point for which the MHD solution has been computed to display parameter exploration.')

#initial value
n_sel=0

if len(selected_points)!=0:

    #only selected the points from the second trace with the curvenumber test
    selected_sol_p_mu=np.array([[selected_points[i]['y'],selected_points[i]['x']] for i in range(len(selected_points)) if selected_points[i]['curveNumber']==1])
    n_sel=len(selected_sol_p_mu)

    if n_sel==0:
        st.warning('None of the selected points have a MHD solution.')

    #fetching the individual solutions with these parameters
    selected_sol_mask=np.array([np.all(np.array([elem[0][2:4] for elem in sol_splitsol])==selected_sol_p_mu[i],axis=1)\
                                for i in range(n_sel)]).any(0)

    selected_mhd_sol=sol_splitsol[selected_sol_mask]

    # ordering the mask in order or reading (increasing mu)

    sol_order=selected_sol_p_mu.T[1].argsort() if sol_order_select =='Increasing mu' else\
                selected_sol_p_mu.T[0].argsort() if sol_order_select == 'Increasing p' else True

    selected_mhd_sol=selected_mhd_sol[sol_order]
    selected_sol_p_mu=selected_sol_p_mu[sol_order]

    sol_p_mhd=selected_sol_p_mu.T[0]

    #doing this without direct transpositions because the arrays arent regular due to uneven angle sampling
    sol_z_over_r=np.array([selected_mhd_sol[i].T[7] for i in range(n_sel)],dtype=object)

    sol_cyl_cst= np.array([np.sqrt(1 + sol_z_over_r[i].astype(float) ** 2) for i in range(n_sel)],dtype=object)


    sol_angle=np.array([selected_mhd_sol[i].T[8] for i in range(n_sel)],dtype=object)

    sol_r_cyl_r0=np.array([selected_mhd_sol[i].T[9] for i in range(n_sel)],dtype=object)
    sol_rho_mhd=np.array([selected_mhd_sol[i].T[10] for i in range(n_sel)],dtype=object)
    sol_t_mhd=np.array([selected_mhd_sol[i].T[14] for i in range(n_sel)],dtype=object)

    sol_ur,sol_uphi,sol_uz=[np.array([selected_mhd_sol[i].T[j] for i in range(n_sel)],dtype=object) for j in range(11,14)]
    sol_br,sol_bphi,sol_bz=[np.array([selected_mhd_sol[i].T[j] for i in range(n_sel)],dtype=object) for j in range(15,18)]

    #start of ideal MHD
    sol_y_id=np.array([selected_mhd_sol[i].T[19][0] for i in range(n_sel)])

    #slow magnetosonic
    sol_y_sm=np.array([selected_mhd_sol[i].T[20][0] for i in range(n_sel)])

    #Alfven
    sol_y_A=np.array([selected_mhd_sol[i].T[21][0] for i in range(n_sel)])


    #converting these last 3 into angles
    def y_to_ang(y):
        '''
        output in degrees
        '''
        return 90-(180/(np.pi)*np.arctan(y))

    sol_angle_id_arr=y_to_ang(sol_y_id)
    sol_angle_sm_arr=y_to_ang(sol_y_sm)
    sol_angle_A_arr=y_to_ang(sol_y_A)

def plotly_line_wrapper(x,y,log_x=False,log_y='auto',xaxis_title='',yaxis_title='',legend=False,
                        line_color='lightskyblue',ex_fig=None,name='',showlegend=False,split_ls_x=None,
                        linedash='solid',figwidth=None,secondary_y=False):

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

    if ex_fig is not None:
        fig_line=ex_fig
    else:
        if secondary_y:
            fig_line=make_subplots(specs=[[{"secondary_y": True}]])
        else:
            fig_line=go.Figure()

        if figwidth is not None:
            fig_line.update_layout(width=figwidth)


    neg_log=False

    if log_x:
        fig_line.update_xaxes(type="log")
    if log_y=='auto':
        #testing if there's no sign change in the y axis
        setup_log_y=(y[y!=0]/abs(y[y!=0])==1).all() or (y[y!=0]/abs(y[y!=0])==-1).all()

        neg_log=(y[y!=0]/abs(y[y!=0])==-1).all()
    elif log_y:
        setup_log_y=True

    if setup_log_y:
        fig_line.update_yaxes(type="log")

    y_title=r'$-'+str(yaxis_title.replace('$',''))+'$' if neg_log else yaxis_title

    if not latex_title:
        #stripping the latex characters in the str:
        x_title_str=xaxis_title.replace('$','').replace('{','').replace('}','').replace('\;',' ').replace("\\",'')\
                    .replace('textrm','').replace('leq','<=').replace('theta','θ')

        y_title_str=y_title.replace('$','').replace('{','').replace('}','').replace('\;',' ').replace("\\",'')\
                    .replace('textrm','').replace('leq','<=').replace('theta','θ')

    else:
        x_title_str=xaxis_title
        y_title_str=y_title

    #note: the <extra> hides the initial template with the name hovering on the left

    if split_ls_x is not None:

        y_split=interp_yaxis(split_ls_x,x,abs(y) if neg_log else y,log_y=setup_log_y)

        #inverted because the angles are backwards

        line_before_split = go.Scatter(x=[split_ls_x]+(x[x<split_ls_x]).tolist(),
                                       y=[y_split]+(abs(y[x<split_ls_x]) if neg_log else y[x<split_ls_x]).tolist(), line=dict(color=line_color, dash=linedash[0]), name=name,
                          showlegend=showlegend,
                          hovertemplate="<b>" + name + "</b><br>" +
                                        x_title_str + ": %{x}<br>" +
                                        y_title_str + ": %{y}<br>" +
                                        "<extra></extra>")

        line_after_split = go.Scatter(x=(x[x>split_ls_x]).tolist()+[split_ls_x],
                                       y=(abs(y[x>split_ls_x]) if neg_log else y[x>split_ls_x]).tolist()+[y_split],
                                      line=dict(color=line_color, dash=linedash[1]), name=name,
                          showlegend=False,
                          hovertemplate="<b>" + name + "</b><br>" +
                                        x_title_str + ": %{x}<br>" +
                                        y_title_str + ": %{y}<br>" +
                                        "<extra></extra>")

        fig_line.add_traces(line_before_split)
        fig_line.add_traces(line_after_split)

    else:
        line=go.Scatter(x=x,y=abs(y) if neg_log else y,line=dict(color=line_color,dash=linedash),name=name,showlegend=showlegend,
                        hovertemplate="<b>"+name+"</b><br>" +
                                      x_title_str+": %{x}<br>" +
                                      y_title_str+": %{y}<br>"+
                                        "<extra></extra>")

        fig_line.add_traces(line)


    fig_line.update_layout(xaxis=dict(showgrid=True,zeroline=False),
                                yaxis=dict(showgrid=True,zeroline=False),
                       xaxis_title=x_title_str, yaxis_title=y_title_str,
                                font=dict(size=14),
                                paper_bgcolor='rgba(0.,0.,0.,0.)', plot_bgcolor='rgba(0.,0.,0.,0.)',
                       margin=dict(l=100, r=20, t=0, b=100))

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

def angle_plot(x_arr,y_arr,log_x=False,log_y='auto',xaxis_title='',yaxis_title='',legend=False,sampl_angles_arr=None,
               compt_angles_arr=None,cmap='cividis',legend_lines=True,legend_points=True):

    fig_line=go.Figure()

    fig_line.update_layout(width=515)

    norm_sol=np.array(range(n_sel))/n_sel

    sol_colors = sample_colorscale(cmap,norm_sol)

    for id,(x,y,sampl_angles,compt_angle,elem_p_mu,sol_angle_id,sol_angle_sm,sol_angle_A,elem_color,elem_sol_angle) in\
            enumerate(zip(x_arr,y_arr,sampl_angles_arr,compt_angles_arr,selected_sol_p_mu,sol_angle_id_arr,sol_angle_sm_arr,sol_angle_A_arr,sol_colors,x_arr)):

        x=x.astype(float)
        y=y.astype(float)
        #adding the line
        line_name='p= '+str(elem_p_mu[0])+' | mu= '+str(elem_p_mu[1])


        #adding a split depending on whether the line is compton thick or not to allow distinguishing the compton
        #thick zones for several solutions


        if n_sel>1:

            # #computing the position of the compton angle on the y axis
            # compt_y=interp_yaxis(compt_angle,x,y)

            fig_line=plotly_line_wrapper(x,y,
                                         log_x=log_x,log_y='auto',
                                         xaxis_title=xaxis_title,yaxis_title=yaxis_title,legend=legend,ex_fig=fig_line,
                                         name=line_name,line_color=elem_color,showlegend=legend_lines,split_ls_x=compt_angle,linedash='solid' if compt_angle is None else ['solid','dash'])

        else:
            fig_line=plotly_line_wrapper(x,y,log_x=log_x,log_y='auto',xaxis_title=xaxis_title,yaxis_title=yaxis_title,legend=legend,ex_fig=fig_line,name=line_name,
                                     line_color=elem_color if n_sel>1 else 'lightskyblue',
                                     showlegend=n_sel>1 and legend_lines)
        neg_log=False

        if log_y=='auto':
            #testing if there's no sign change in the y axis
            setup_log_y=(y[y!=0]/abs(y[y!=0])==1).all() or (y[y!=0]/abs(y[y!=0])==-1).all()

            neg_log=(y[y!=0]/abs(y[y!=0])==-1).all()
        elif log_y:
            setup_log_y=True

        #adding the specific points by interpolating
        #note: we only show the legend in the last iteration to have the symbols labels below the line labels

        y_title = r'$-' + str(yaxis_title.replace('$', '')) + '$' if neg_log else yaxis_title

        if not latex_title:
            # stripping the latex characters in the str:
            x_title_str = xaxis_title.replace('$', '').replace('{', '').replace('}', '').replace('\;', ' ').replace(
                "\\", '') \
                .replace('textrm', '').replace('leq', '<=').replace('theta', 'θ')

            y_title_str = y_title.replace('$', '').replace('{', '').replace('}', '').replace('\;', ' ').replace("\\",
                                                                                                                '') \
                .replace('textrm', '').replace('leq', '<=').replace('theta', 'θ')

        else:
            x_title_str = xaxis_title
            y_title_str = y_title

        id_mhd_point=go.Scatter(x=[sol_angle_id],y=[(-1 if neg_log else 1)*\
                                interp_yaxis(sol_angle_id,x,y,log_y=setup_log_y,log_x=log_x)],mode='markers',name='id mhd point',marker=dict(size=13,symbol='star-triangle-up-dot',color='black',line=dict(color='violet',width=2)),showlegend=id==n_sel-1 and legend_points,
                                hovertemplate="<b>" + 'id mhd point' + "</b><br>" +
                                              "<b>" + line_name + "</b><br>"+
                                              x_title_str + ": %{x}<br>" +
                                              y_title_str + ": %{y}<br>" +
                                              "<extra></extra>")

        sm_point = go.Scatter(x=[sol_angle_sm], y=[(-1 if neg_log else 1)*\
                                interp_yaxis(sol_angle_sm, x,y,log_y=setup_log_y,log_x=log_x)], mode='markers', name='sm point',marker=dict(size=13,symbol='star-triangle-down-dot',color='black',line=dict(color='green',width=2)),showlegend=id==n_sel-1 and legend_points,
                              hovertemplate="<b>" + 'sm point' + "</b><br>" +
                                            "<b>" + line_name + "</b><br>"+
                                            x_title_str + ": %{x}<br>" +
                                            y_title_str + ": %{y}<br>" +
                                            "<extra></extra>")

        Alfven_point = go.Scatter(x=[sol_angle_A],y=[(-1 if neg_log else 1)*\
                                interp_yaxis(sol_angle_A,x,y,log_y=setup_log_y,log_x=log_x)], mode='markers', name='Alfven point',marker=dict(size=13,symbol='circle-x',color='black',line=dict(color='red',width=2)),showlegend=id==n_sel-1 and legend_points,
                                  hovertemplate="<b>" + 'Alfven point' + "</b><br>" +
                                                "<b>" + line_name + "</b><br>"+
                                                x_title_str + ": %{x}<br>" +
                                                y_title_str + ": %{y}<br>" +
                                                "<extra></extra>")

        fig_line.add_traces(id_mhd_point)
        fig_line.add_traces(sm_point)
        fig_line.add_traces(Alfven_point)

        #adding the angle sampling

        if sampl_angles is not None:

            mask_sampl_angles=np.array([elem_sol_angle==elem for elem in sampl_angles]).any(0)

            sampl_points=go.Scatter(x=x[mask_sampl_angles],y=(-1 if neg_log else 1)*y[mask_sampl_angles], mode='markers', name='angle sampling',marker=dict(size=10,symbol='line-ns',color='black',line=dict(color='orange',width=2)),showlegend=id==n_sel-1 and legend_points,
                                    hovertemplate="<b>" + 'angle sampling' + "</b><br>" +
                                                  "<b>" + line_name + "</b><br>" +
                                                  x_title_str + ": %{x}<br>" +
                                                  y_title_str + ": %{y}<br>" +
                                                  "<extra></extra>")
            fig_line.add_traces(sampl_points)

        #and the compton thick region for the single lines
        if compt_angle:
            if  n_sel==1:
                fig_line.add_vrect(x0=compt_angle, x1=90, line_width=0, fillcolor="grey", opacity=0.2,
                                   annotation_text="compton-thick", annotation_position="bottom",
                                   annotation=dict(font=dict(color='white')))
            else:
                #remaking this because we need it
                if log_y == 'auto':
                    # testing if there's no sign change in the y axis
                    setup_log_y = (y[y != 0] / abs(y[y != 0]) == 1).all() or (y[y != 0] / abs(y[y != 0]) == -1).all()

                    neg_log = (y[y != 0] / abs(y[y != 0]) == -1).all()
                elif log_y:
                    setup_log_y = True

                compt_angle_y=interp_yaxis(compt_angle,x,abs(y) if neg_log else y,log_y=setup_log_y)

                #marking the beginning of the compton thick region with a marker

                compt_thick_point= go.Scatter(x=[compt_angle], y=[compt_angle_y], mode='markers', name='compton thick thresh',marker=dict(size=13,symbol='triangle-right',color='black',line=dict(color='white',width=2)),showlegend=id==n_sel-1 and legend_points,
                                      hovertemplate="<b>" + 'compton thick thresh' + "</b><br>" +
                                                    "<b>" + line_name + "</b><br>"+
                                                    x_title_str + ": %{x}<br>" +
                                                    y_title_str + ": %{y}<br>" +
                                                    "<extra></extra>")

                fig_line.add_trace(compt_thick_point)

    if latex_title:
        st.components.v1.html(fig_line.to_html(include_mathjax='cdn'), width=500, height=450)
    else:
        st.plotly_chart(fig_line,use_container_width=False,theme=None)



if split_angle and n_sel>0:

    array_sampl_angle,compton_angles_arr=sample_angle(solutions_path,
                                          angle_values=np.arange(val_angle_low,val_angle_high+0.00001,val_angle_step),
                                          mdot_obs=mdot_obs,m_BH=m_BH,r_j=rj,mode='array',return_compton_angle=True)

    sol_split_angle=load_solutions(array_sampl_angle,mode='array',split_sol=True,split_par=False)

    selected_sol_split_angle=sol_split_angle[selected_sol_mask]

    #re-ordering
    selected_sol_split_angle=selected_sol_split_angle[sol_order]

    selected_angles=[selected_sol_split_angle[i].T[8] for i in range(n_sel)]

    compton_angles=compton_angles_arr[selected_sol_mask]

    compton_angles=compton_angles[sol_order]

else:
    selected_angles=[None]*n_sel
    compton_angles=[None]*n_sel

with tab_sol:

    #only showing a title with the solution parameters if there's a single one, otherwise we put legends
    if n_sel==1:
        st.title(r'solution: $\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\textrm{p}='+str(round(selected_sol_p_mu[0][0],4))+
                 '\;\;\;|\;\;\; \mu='+str(round(selected_sol_p_mu[0][1],4))+'$')

    #now we can display the plots of the individual solution
    col_1,col_2,col_3=st.columns(3)

    if n_sel>0:
        with col_1:

            angle_plot(sol_angle,sol_r_cyl_r0,log_x=False,log_y=True,legend_points=n_sel==1,
                                xaxis_title=r'$\theta \; (°)$',yaxis_title=r'$r_{cyl}/r_0$',legend=True,
                                sampl_angles_arr=selected_angles,compt_angles_arr=compton_angles)

            angle_plot(sol_angle,sol_ur,log_x=False,legend_points=n_sel==1,
                                xaxis_title=r'$\theta \; (°)$', yaxis_title=r'$u_{r}$',legend=True,
                                sampl_angles_arr=selected_angles,compt_angles_arr=compton_angles)

            angle_plot(sol_angle,sol_br,log_x=False,legend_points=n_sel==1,
                                xaxis_title=r'$\theta \; (°)$', yaxis_title=r'$B_r$',legend='top_left',
                                sampl_angles_arr=selected_angles,compt_angles_arr=compton_angles)

        with col_2:

            #when several solutions are selected we plot the legend for the points here to avoid cluttering the first
            #plots
            angle_plot(sol_angle,sol_rho_mhd,log_x=False,log_y=True,
                       legend='top_left' if n_sel>1 else False,legend_lines=False,
                                xaxis_title =r'$\theta \; (°)$', yaxis_title = r'$\ρ_{mhd}$',
                                sampl_angles_arr=selected_angles,compt_angles_arr=compton_angles)

            angle_plot(sol_angle, sol_uphi,log_x=False,log_y=True,
                       legend='top_left' if n_sel>1 else False,legend_lines=False,
                                xaxis_title=r'$\theta \; (°)$', yaxis_title=r'$u_{\phi}$',
                                sampl_angles_arr=selected_angles,compt_angles_arr=compton_angles)

            angle_plot(sol_angle, sol_bphi,log_x=False,legend='bot_left' if n_sel>1 else False,legend_lines=False,
                                xaxis_title=r'$\theta \; (°)$', yaxis_title=r'$B_{\phi}$',
                                sampl_angles_arr=selected_angles,compt_angles_arr=compton_angles)


        with col_3:

            angle_plot(sol_angle,sol_t_mhd,log_x=False,log_y=True,
                                xaxis_title=r'$\theta \; (°)$', yaxis_title=r'$T_{mhd}$',
                                sampl_angles_arr=selected_angles,compt_angles_arr=compton_angles)

            angle_plot(sol_angle, sol_uz,log_x=False,
                                xaxis_title=r'$\theta \; (°)$', yaxis_title=r'$u_{z}$',
                                sampl_angles_arr=selected_angles,compt_angles_arr=compton_angles)

            angle_plot(sol_angle, sol_bz,log_x=False,
                                xaxis_title=r'$\theta \; (°)$', yaxis_title=r'$B_z$',
                                sampl_angles_arr=selected_angles,compt_angles_arr=compton_angles)

if n_sel>1:
    with tab_sol_radial:
        st.info('Display of radial evolution restricted to single solution selection.')
    with tab_explo:
        st.info('Parameter exploration restricted to single solution selection.')

if n_sel==1 and not split_angle:
    with tab_sol_radial:
        st.info('Activate angle sampling in the sidebar to see radial distributions.')
    with tab_explo:
        st.info('Activate angle sampling in the sidebar to see parameter exploration.')

def radial_plot(rad,sol_sampl,angl_sampl,log_x=False,log_y=False,xaxis_title='',yaxis_title='',legend=False,
                cmap='plasma_r',logxi_ids=None,yrange=None):

    norm_angl= (angl_sampl-val_angle_low)/(val_angle_high-val_angle_low)

    ang_colors = sample_colorscale(cmap,norm_angl)

    #creating the theme with the first line
    fig_rad=plotly_line_wrapper(rad[0],sol_sampl[0],log_x=log_x,log_y='auto',xaxis_title=xaxis_title,yaxis_title=yaxis_title,line_color=ang_colors[0],legend=True,name='θ='+str(angl_sampl[0]))

    fig_rad.update_layout(width=515)

    neg_log=False

    y_title = r'$-' + str(yaxis_title.replace('$', '')) + '$' if neg_log else yaxis_title

    if not latex_title:
        # stripping the latex characters in the str:
        x_title_str = xaxis_title.replace('$', '').replace('{', '').replace('}', '').replace('\;', ' ').replace(
            "\\", '') \
            .replace('textrm', '').replace('leq', '<=').replace('theta', 'θ')

        y_title_str = y_title.replace('$', '').replace('{', '').replace('}', '').replace('\;', ' ').replace("\\",
                                                                                                            '') \
            .replace('textrm', '').replace('leq', '<=').replace('theta', 'θ')

    else:
        x_title_str = xaxis_title
        y_title_str = y_title

    #and adding the rest of the lines
    for id_sol,(elem_rad,elem_sampl,elem_angl,elem_color) in enumerate(zip(rad[1:],sol_sampl[1:],angl_sampl[1:],ang_colors[1:])):
        fig_rad.add_trace(
        go.Scatter(x=elem_rad,y=elem_sampl,line=dict(color=elem_color),name='',showlegend=False,
            hovertemplate="<b>" + 'θ='+str(elem_angl) + "</b><br>" +
                          x_title_str + ": %{x}<br>" +
                          y_title_str + ": %{y}<br>" +
                          "<extra></extra>"))

    #adding the trace of the logxi=6 surface
    if logxi_ids is not None:
        fig_rad.add_trace(go.Scatter(x=[rad[i][logxi_ids[i]] for i in range(len(sol_sampl))],
                                     y=[sol_sampl[i][logxi_ids[i]] for i in range(len(sol_sampl))],
                                     name='logxi=6',mode='lines',line=dict(color='white')))

    #tickvals for the cmap with some rounding
    tickvals_cm=np.array([round(elem,1) for elem in angl_sampl.tolist()+[val_angle_low,val_angle_high]])

    colorbar_trace = go.Scatter(x=[None],y=[None],mode='markers',showlegend=False,
                                marker=dict(colorscale=cmap,showscale=True,
                                            cmin=val_angle_low,cmax=val_angle_high,
                                    colorbar=dict(thickness=10, tickvals=tickvals_cm,tickfont=dict(color='white'),
                                                  ticks='outside',ticklen=3,tickcolor='white',
                                                  title=dict(text='θ' if not latex_title else r"theta", side="top"),titlefont=dict(color='white'))),hoverinfo='none')
    fig_rad.add_trace(colorbar_trace)

    fig_rad.update_layout(xaxis=dict(range=[np.log10(rj),6]))

    if yrange is not None:
        fig_rad.update_layout(yaxis=dict(range=yrange))

    if latex_title:
        st.components.v1.html(fig_rad.to_html(include_mathjax='cdn'), width=500, height=450)
    else:
        st.plotly_chart(fig_rad, use_container_width=False, theme=None)


mdot_mhd=mdot_obs*12

if split_angle and n_sel==1:
    n_sol=len(selected_sol_split_angle[0])

    sol_sampl_z_over_r = selected_sol_split_angle[0].T[7]
    sol_sampl_angle = selected_sol_split_angle[0].T[8]

    sol_sampl_r_cyl_r0 = selected_sol_split_angle[0].T[9]
    sol_sampl_rho_mhd = selected_sol_split_angle[0].T[10]

    sol_sampl_ur, sol_sampl_uphi, sol_sampl_uz = selected_sol_split_angle[0].T[11:14]
    sol_sampl_br, sol_sampl_bphi, sol_sampl_bz = selected_sol_split_angle[0].T[15:18]

    sol_p_mhd=selected_sol_split_angle[0][0][3]

    cyl_cst_sampl=np.sqrt(1+sol_sampl_z_over_r**2)

    r_sph_sampl=np.array([np.logspace(np.log10(rj*cyl_cst_sampl[i]),7,300) for i in range(n_sol)])

    with tab_sol_radial:

        n_sampl=np.array([func_density_sol(r_sph_sampl[i],sol_sampl_z_over_r[i],sol_sampl_rho_mhd[i],sol_p_mhd,
                                           mdot_mhd,m_BH) for i in range(n_sol)])

        logxi_sampl=np.array([func_logxi_sol(r_sph_sampl[i],sol_sampl_z_over_r[i],val_L_source,sol_sampl_rho_mhd[i],
                                             sol_p_mhd,mdot_mhd,sol_sampl_ur[i], sol_sampl_uphi[i],
                                             sol_sampl_uz[i],m_BH)\
                              for i in range(n_sol)])

        nh_sampl=np.array([func_nh_sol(r_sph_sampl[i],rj*cyl_cst_sampl[i],sol_sampl_z_over_r[i],
                                       sol_sampl_rho_mhd[i],sol_p_mhd,mdot_mhd,m_BH) for i in range(n_sol)])

        #here all speeds are divided by 1e5 to go to km/s
        ur_sampl=np.array([func_vel_sol('r',r_sph_sampl[i],sol_sampl_z_over_r[i],sol_sampl_ur[i],
                                        sol_sampl_uphi[i],sol_sampl_uz[i],m_BH) for i in range(n_sol)])/1e5

        uphi_sampl=np.array([func_vel_sol('phi',r_sph_sampl[i],sol_sampl_z_over_r[i],sol_sampl_ur[i],
                                          sol_sampl_uphi[i],sol_sampl_uz[i],m_BH) for i in range(n_sol)])/1e5

        uz_sampl=np.array([func_vel_sol('z',r_sph_sampl[i],sol_sampl_z_over_r[i],sol_sampl_ur[i],
                                        sol_sampl_uphi[i],sol_sampl_uz[i],m_BH) for i in range(n_sol)])/1e5

        uobs_sampl=np.array([func_vel_sol('obs',r_sph_sampl[i],sol_sampl_z_over_r[i],sol_sampl_ur[i],
                                          sol_sampl_uphi[i],sol_sampl_uz[i],m_BH) for i in range(n_sol)])/1e5

        #fetching the positions at which logxi=6 for each angle
        logxi_6_ids=np.array([np.argmin(abs(elem-6)) for elem in logxi_sampl])

        r_sph_nonthick_sampl=np.array([r_sph_sampl[i][logxi_6_ids[i]:] for i in range(n_sol)],dtype=object)

        nh_nonthick_sampl=np.array([func_nh_sol(r_sph_sampl[i][logxi_6_ids[i]:],r_sph_sampl[i][logxi_6_ids[i]],
                            sol_sampl_z_over_r[i],sol_sampl_rho_mhd[i],sol_p_mhd,mdot_mhd,m_BH) for i in range(n_sol)],
                                   dtype=object)

        col_a,col_b,col_c=st.columns(3)

        yrange_speed=[np.log10(1),np.log10(299792.458)]

        with col_a:
            radial_plot(r_sph_sampl,n_sampl,sol_sampl_angle,log_x=True,
                                        log_y=True,xaxis_title=r'$R_{sph}\;$ (Rg)',yaxis_title=r'$n\textrm{ (cgs)}$',
                        logxi_ids=logxi_6_ids)

            radial_plot(r_sph_sampl,ur_sampl,sol_sampl_angle,log_x=True,
                                        log_y=True,xaxis_title=r'$R_{sph}\;$ (Rg)',yaxis_title=r'$v_{r}\textrm{ (km/s)}$',
                        logxi_ids=logxi_6_ids,yrange=yrange_speed)

            radial_plot(r_sph_sampl,nh_sampl,sol_sampl_angle,log_x=True,log_y=True,
                        xaxis_title=r'$R_{sph}\;$ (Rg)',yaxis_title=r'$n_{h}\textrm{ (cm}^{-2}\textrm{)}$',
                        logxi_ids=logxi_6_ids,
                        yrange=[np.log10(1e20),np.log10(max(ravel_ragged(nh_sampl)))])

        with col_b:
            radial_plot(r_sph_sampl,logxi_sampl, sol_sampl_angle, log_x=True,
                        log_y=False, xaxis_title=r'$R_{sph}\;$ (Rg)', yaxis_title=r'$log\xi$',
                        logxi_ids=logxi_6_ids)

            radial_plot(r_sph_sampl, uphi_sampl, sol_sampl_angle, log_x=True,
                        log_y=True, xaxis_title=r'$R_{sph}\;$ (Rg)', yaxis_title=r'$v_{\phi}\textrm{ (km/s)}$',
                        logxi_ids=logxi_6_ids,yrange=yrange_speed)

            radial_plot(r_sph_nonthick_sampl,nh_nonthick_sampl,sol_sampl_angle,log_x=True,log_y=True,
                        xaxis_title=r'$R_{sph}\;$ (Rg)',yaxis_title=r'$n_{h}^{\textrm{log}\xi\leq6}\textrm{(cm}^{-2}\textrm{)}$',
                        yrange=[np.log10(1e20),np.log10(max(ravel_ragged(nh_nonthick_sampl)))])

        with col_c:

            radial_plot(r_sph_sampl, uobs_sampl, sol_sampl_angle, log_x=True,
                        log_y=True, xaxis_title=r'$R_{sph}\;$ (Rg)', yaxis_title=r'$v_{obs}\textrm{ (km/s)}$',
                        logxi_ids=logxi_6_ids,yrange=yrange_speed)

            radial_plot(r_sph_sampl, uz_sampl, sol_sampl_angle, log_x=True,
                        log_y=True, xaxis_title=r'$R_{sph}\;$ (Rg)', yaxis_title=r'$v_{z}\textrm{ (km/s)}$',
                        logxi_ids=logxi_6_ids,yrange=yrange_speed)

    #luminosity at which logxi is 6

    m_BH_SI = m_BH * Msol_SI
    Rs_SI = 2.0 * G_SI * m_BH_SI / (c_SI * c_SI)

    # !* Gravitational radius
    Rg_SI = 0.5 * Rs_SI
    Rg_cgs = Rg_SI * m2cm

    L_xi_6=10**6*n_sampl*Rg_cgs**2*r_sph_sampl**2

    L_xi_6_Edd=L_xi_6/(1.26e38*m_BH)

    with tab_explo:

        col_explo_a,col_explo_b,col_explo_c=st.columns(3)

        with col_explo_a:
            radial_plot(r_sph_sampl,L_xi_6, sol_sampl_angle, log_x=True,
                        log_y=True, xaxis_title=r'$R_{sph}\;$ (Rg)', yaxis_title=r'L (cgs)$')


# def polar2cartesian(rad_range, theta_range, grid, x, y, order=3):
#
#     #taken from https://stackoverflow.com/questions/2164570/reprojecting-polar-to-cartesian-grid
#
#     X, Y = np.meshgrid(x, y)
#
#     new_r = np.sqrt(X * X + Y * Y)
#     new_t = np.arctan2(X, Y)
#
#     ir = interp1d(rad_range, np.arange(len(rad_range)), bounds_error=False)
#     it = interp1d(theta_range, np.arange(len(theta_range)), bounds_error=False)
#
#     new_ir = ir(new_r.ravel())
#
#     new_it = it(new_t.ravel())
#
#     breakpoint()
#
#     new_ir[new_r.ravel() > rad_range.max()] = len(rad_range) - 1
#     new_ir[new_r.ravel() < rad_range.min()] = 0
#
#     return map_coordinates(grid, np.array([new_ir, new_it]),
#                            order=order).reshape(new_r.shape)

import cv2

#2D plotting function
def plot_2D(r_sampl_sol,angle_sampl_sol,data,r_j,r_max,n_rad,cmap='plasma',log_sampl=True,figwidth=515):


    #interpolating back onto a cartesian grid
    coord_sampl=np.logspace(np.log10(r_j),np.log10(r_max),n_rad*3) if log_sampl else np.linspace(r_j,r_max,n_rad)

    #polar_img = cv2.warpPolar(image, (256, 1024), (image.shape[0] / 2, image.shape[1] / 2),
     #                         image.shape[1] * margin * 0.5, cv2.WARP_POLAR_LINEAR)


    #initial values grid
    coord_cyl=np.array([[r_sampl_sol[i]*np.cos(angle_sampl_sol[j]*np.pi/180),
                         r_sampl_sol[i]*np.sin(angle_sampl_sol[j]*np.pi/180)]
                         for i in range(len(r_sampl_sol)) for j in range(len(angle_sampl_sol))])

    coord_cart=np.array([np.repeat(coord_sampl,len(coord_sampl)).reshape(len(coord_sampl),len(coord_sampl)),
                         np.repeat(coord_sampl,len(coord_sampl)).reshape(len(coord_sampl),len(coord_sampl)).T])



    cart_data=scipy.interpolate.griddata(coord_cyl, np.ravel(data),(coord_cart[0],coord_cart[1]),
                                         method='linear', fill_value=0, rescale=False)

    #cart_data = cv2.warpPolar(np.log10(data), (np.shape(data)[1],np.shape(data)[0]), (0,0),n_rad,
     #                         flags=cv2.WARP_INVERSE_MAP+cv2.WARP_POLAR_LOG)


    #cart_data=polar2cartesian(r_sampl_sol,angle_sampl_sol.astype(float)*np.pi/180,np.log10(data),coord_sampl,coord_sampl,order=3)

    fig_2D = go.Figure()

    if figwidth is not None:
        fig_2D.update_layout(width=figwidth)
        fig_2D.update_layout(height=figwidth)
    if log_sampl:

        fig_2D.update_xaxes(type="log")
        fig_2D.update_yaxes(type="log")

    trace_2D=go.Heatmap(x=coord_sampl,
            y=coord_sampl,
            z=np.log10(cart_data),
            zsmooth='best',
            type='heatmap',
            colorscale=cmap)

    fig_2D.add_trace(trace_2D)

    return fig_2D

with tab_2D:

    if n_sel==1 and visu_2D:

        '''
        Creating the full 2D mapping for the single solution to do quarter angle plotting
        '''

        n_rad=300

        n_angles_sol_indiv=len(sol_z_over_r[0])


        #r_sph_sol_indiv = np.array([np.logspace(np.log10(rj * cyl_cst_sol_indiv[i]), 7, n_rad) for i in range(n_angles_sol_indiv)])

        # using a constant start value for now to avoid issues when interpolating back to cartesian grid
        r_sph_sol= np.logspace(np.log10(rj), 7, n_rad)

        #see to replace the singular values with arrays if we use several solutions at once

        n_sol_map= np.array([[func_density_sol(r_sph_sol, sol_z_over_r[i][j], sol_rho_mhd[i][j], sol_p_mhd,
                                             mdot_mhd, m_BH) for j in range(len(sol_z_over_r[i]))] for i in range(n_sel)])

        # logxi_sol_map = np.array([func_logxi_sol(r_sph_sol, sol_z_over_r[0][i], val_L_source, sol_rho_mhd[0][i],
        #                                        sol_p_mhd, mdot_mhd, m_BH) for i in range(n_angles_sol_indiv)])
        #
        # nh_sol_map = np.array([func_nh_sol(r_sph_sol_indiv[i], rj * cyl_cst_sol_indiv[i], sol_z_over_r[0][i],
        #                                  sol_rho_mhd[0][i], sol_p_mhd, mdot_mhd, m_BH) for i in range(n_angles_sol_indiv)])
        #
        # # here all speeds are divided by 1e5 to go to km/s
        # ur_sol_indiv = np.array([func_vel_sol('r', r_sph_sol_indiv[i], sol_z_over_r[0][i], sol_ur[0][i],
        #                                   sol_uphi[0][i], sol_uz[0][i], m_BH) for i in range(n_angles_sol_indiv)]) / 1e5
        #
        # uphi_sol_indiv = np.array([func_vel_sol('phi', r_sph_sol_indiv[i], sol_z_over_r[0][i], sol_ur[0][i],
        #                                     sol_uphi[0][i], sol_uz[0][i], m_BH) for i in range(n_angles_sol_indiv)]) / 1e5
        #
        # uz_sol_indiv = np.array([func_vel_sol('z', r_sph_sol_indiv[i], sol_z_over_r[0][i], sol_ur[0][i],
        #                                   sol_uphi[0][i], sol_uz[0][i], m_BH) for i in range(n_angles_sol_indiv)]) / 1e5
        #
        # uobs_sol_indiv = np.array([func_vel_sol('obs', r_sph_sol_indiv[i], sol_z_over_r[0][i], sol_ur[0][i],
        #                                     sol_uphi[0][i], sol_uz[0][i], m_BH) for i in range(n_angles_sol_indiv)]) / 1e5
        #
        # # fetching the positions at which logxi=6 for each angle
        # logxi_6_ids = np.array([np.argmin(abs(elem - 6)) for elem in logxi_sol_indiv])
        #
        # r_sph_nonthick_sol_indiv = np.array([r_sph_sol_indiv[i][logxi_6_ids[i]:] for i in range(n_angles_sol_indiv)], dtype=object)
        #
        # nh_nonthick_sol_indiv = np.array([func_nh_sol(r_sph_sol_indiv[i][logxi_6_ids[i]:], r_sph_sol_indiv[i][logxi_6_ids[i]],
        #                                           sol_z_over_r[0][i], sol_rho_mhd[0][i], sol_p_mhd, mdot_mhd, m_BH)
        #                               for i in range(n_angles_sol_indiv)],
        #                              dtype=object)


        test=plot_2D(r_sph_sol, sol_angle[0],n_sol_map[0].T,r_j=rj,r_max=1e7,n_rad=n_rad)

        import matplotlib.pyplot as plt

        fig,ax=plt.subplots(1,1,subplot_kw=dict(polar=True))


        mesh=ax.pcolormesh(np.pi/2-sol_angle[0].astype(float)*np.pi/180, r_sph_sol, np.log10(n_sol_map[0].T),shading='nearest',
                      cmap='plasma')
        ax.grid(alpha=0.1)
        ax.set_rlim(0)
        ax.set_rscale('symlog')
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        fig.colorbar(mesh, ax=ax)
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.pyplot(fig)

        with col_b:
            st.plotly_chart(test)

        with col_c:
            plt.figure()
            test1=plt.imshow(np.log10(n_sol_map[0].T),cmap='plasma')
            fig1=test1.get_figure()

            st.pyplot(fig1)
#thermal structure and aspect ratio (everything should be in cgs, except r in Rg units)

#opacity regimes in decreasing temperature relevancy
#Kr=K_0*rho^(alpha)*T_c^(Beta)
# in cgs, from Bell & Lin 1994 and Frank, King and Raine 2002 for the Kramer values

#electrons
kappa_0_e=0.348
alpha_tau_e=0
beta_tau_e=0

#Kramer (only Free-Free)
kappa_0_kramer=5e24
alpha_tau_kramer=1
beta_tau_kramer=-7/2

#Bound-Free and Free/Free
#deactivated for tests
#kappa_0_BFFF=1.5e20
kappa_0_BFFF=1.5e20
alpha_tau_BFFF=1
beta_tau_BFFF=-5/2

#Hydrogen scattering
kappa_0_H=1e-36
alpha_tau_H=1/3
beta_tau_H=10

#Molecules
#real value kappa_0_mol=1e-8
kappa_0_mol=1e-80
alpha_tau_mol=2/3
beta_tau_mol=3

#arrays
kappa_0_arr=np.array([kappa_0_e,kappa_0_kramer,kappa_0_BFFF,kappa_0_H,kappa_0_mol])
alpha_tau_arr=np.array([alpha_tau_e,alpha_tau_kramer,alpha_tau_BFFF,alpha_tau_H,alpha_tau_mol])
beta_tau_arr=np.array([beta_tau_e,beta_tau_kramer,beta_tau_BFFF,beta_tau_H,beta_tau_mol])

n_regimes=len(kappa_0_arr)

name_regimes=['electrons','Kramer','B-F/F-F','Hydrogen','Molecules']

norm_col_regimes=np.array(range(n_regimes))/n_regimes

col_regimes=sample_colorscale('plasma',norm_col_regimes)

#function for the H/R ratio

def func_H_R(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta=0.1,mu=0.1,b=0,p=0,alpha_0=10,m_BH=1,r_j=6,
             mode='interp'):

    '''
    wrapper for the H_R (or epsilon) analytical solution for a standard optically thick SAD

    modes:
    -gaz: general solution (any opacity regime) when gaz pressure dominates
    -rad: general solution (any opacity regime) when radiative pressure dominates
    -interp: computes the radiation and gaz pressure in gaz mode,
             then replaces the zones in rad mode with the computation in rad mode
             (ad-hoc but works independantly of the opacity regime)
             then, recomputes Prad and Pgaz and ensures Prad is still > Pgaz in the Prad dominated zone
             (else returns an error)



    r: radius in Rg

    mdot_in:mass accretion rate at the starting radius

    kappa_0,alpha_tau,beta_tau: coefficients to compute the opacity (see above)

    mu: magnetization

    b: Jet Power index
    p: ejection index

    both default to 0

    p is constant along r, but if b is not constant the code should be edited

    alpha_0= viscosity coefficient, can also be seen as alpha_m*Prandlt_m, value used with mu to get the
                        viscosity parameter of the disk

    m_BH: Black Hole mass in solar masses
    '''

    #defining useful mass dependant constants
    m_BH_cgs=m_BH*Msol_cgs
    Rg_cgs=G_cgs*m_BH_cgs/c_cgs**2
    n_star=1/(sigma_thomson_cgs*Rg_cgs)
    L_Edd = 1.26e38 * m_BH

    M_Edd= L_Edd / c_cgs ** 2

    # radial dependant mdot in CGS units
    mdot_r = (r_arr/r_j) ** (p) * mdot_in

    #value for Epsilon with the powers of the opacity still in in P_gaz regime

    def H_R_gaz_dom():
        H_R_expr = (m_p_cgs * c_cgs ** 2 / (2 * k_boltzmann_cgs)) ** (beta_tau - 4) \
                   * (m_p_cgs * n_star / (alpha_0 * mu ** (1 / 2))) ** (alpha_tau + 1) \
                   * (3 * kappa_0 / (4 * sigma_boltzmann_cgs)) * (1 - b) * (1 - p) \
                   * (c_cgs ** 4 / (8 * np.pi * G_cgs * m_BH_cgs)) \
                   * (mdot_r) ** (alpha_tau + 2) * M_Edd \
                   * r_arr ** (4 - beta_tau - 3 / 2 * (alpha_tau + 1) - 2) \
 \
            # we don't simplify the last exponent to make verifications easier
        return H_R_expr ** (1 / (10 + 3 * alpha_tau - 2 * beta_tau))

    def H_R_rad_dom():
        H_R_expr = (m_p_cgs * n_star / (alpha_0 * mu ** (1 / 2))) ** (-(alpha_tau + 1)) \
                   * (3 * c_cgs ** 3 * m_p_cgs * n_star / (4 * sigma_boltzmann_cgs * alpha_0 * mu ** (1 / 2))) ** ( -b / 4) \
                   * 2 / ((1 - b) * (1 - p) * kappa_0 * Rg_cgs * alpha_0 * mu ** (1 / 2)) \
                   * mdot_r ** (-(alpha_tau + 1 + beta_tau / 4)) \
                   * r_arr ** (3 / 2 * (alpha_tau + 1) + 5 * beta_tau / 8 - 1 / 2)
        # we don't simplify the last exponent to make verifications easier
        return H_R_expr ** (1 / (-3 * (alpha_tau + 1) - beta_tau / 4 + 2))

    if mode=='gaz':
        return H_R_gaz_dom()
    elif mode=='rad':
        return H_R_rad_dom()
    elif mode=='bridge':
        P_gaz_gaz = func_P_gaz(r_arr, mdot_in, kappa_0, alpha_tau, beta_tau, eta, mu, b, p, alpha_0, m_BH, r_j,
                               mode='gaz')
        P_rad_gaz=func_P_rad(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j,
                             mode='gaz')

        P_rad_dom_mask=P_rad_gaz>P_gaz_gaz

        H_R_gaz=H_R_gaz_dom()
        H_R_rad=H_R_rad_dom()

        H_R_bridge=np.array([H_R_rad[i] if P_rad_dom_mask[i] else H_R_gaz[i] for i in range(len(r_arr))])

        return H_R_bridge

def func_rho_0(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta=0.1,mu=0.1,b=0,p=0,alpha_0=10,m_BH=1,r_j=6,mode='bridge'):

    '''
    wrapper for the rho_0 (density at midplane) solution for a standard SAD in the P_gaz dominant regime

    r: radius in Rg

    mdot_in:mass accretion rate at the starting radius

    kappa_0,alpha_tau,beta_tau: coefficients to compute the opacity (see above)

    mu: magnetization

    b: Jet Power index
    p: ejection index

    both default to 0

    p is constant along r, but if b is not constant the code should be edited

    alpha_0= viscosity coefficient, can also be seen as alpha_m*Prandlt_m, value used with mu to get the
                        viscosity parameter of the disk

    m_BH: Black Hole mass in solar masses
    '''

    #defining useful mass dependant constants
    m_BH_cgs=m_BH*Msol_cgs
    Rg_cgs=G_cgs*m_BH_cgs/c_cgs**2
    n_star=1/(sigma_thomson_cgs*Rg_cgs)
    L_Edd = 1.26e38 * m_BH

    M_Edd= L_Edd / c_cgs ** 2

    # radial dependant mdot in CGS units
    mdot_r = (r_arr/r_j) ** (p) * mdot_in

    rho_0=m_p_cgs*n_star * 1/(alpha_0*mu**(1/2)) * mdot_r * r_arr ** (-3/2) *\
          func_H_R(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j,mode=mode)**(-3)

    # rho_0=m_p_cgs*n_star * 1/(alpha_0*mu**(1/2)) * mdot_r * r**(-3/2) * 0.1**(-3)

    #*L_Edd / c_cgs ** 2

    return rho_0

def func_T_0(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta=0.1,mu=0.1,b=0,p=0,alpha_0=10,m_BH=1,r_j=6,mode='bridge'):

    '''
    wrapper for the T_0 (temperature at midplane) solution for a standard SAD in the P_gaz dominant regime

    r: radius in Rg

    mdot_in:mass accretion rate at the starting radius

    kappa_0,alpha_tau,beta_tau: coefficients to compute the opacity (see above)

    mu: magnetization

    b: Jet Power index
    p: ejection index

    both default to 0

    p is constant along r, but if b is not constant the code should be edited

    alpha_0= viscosity coefficient, can also be seen as alpha_m*Prandlt_m, value used with mu to get the
                        viscosity parameter of the disk

    m_BH: Black Hole mass in solar masses
    '''
    #defining useful mass dependant constants
    m_BH_cgs=m_BH*Msol_cgs
    Rg_cgs=G_cgs*m_BH_cgs/c_cgs**2
    n_star=1/(sigma_thomson_cgs*Rg_cgs)
    L_Edd = 1.26e38 * m_BH

    M_Edd= L_Edd / c_cgs ** 2

    # radial dependant mdot in CGS units
    mdot_r = (r_arr/r_j) ** (p) * mdot_in

    # T_0=m_p_cgs*c_cgs**2/(2*k_boltzmann_cgs) *\
    #     1/r_arr * func_H_R(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j)**2

    #this one should work no matter the pressure as long as no advection

    def T0_rad_dom():

        #this only work in Prad and is no good for bridging
        T_0=(3*c_cgs**3*m_p_cgs*n_star/(4*sigma_thomson_cgs*alpha_0*mu**(1/2))*mdot_r*r_arr**(-5/2) \
            *func_H_R(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j,mode=mode)**(-1))**(1/4)
        return T_0

    def T0_gen():

        #this one has been computed directly from the qrad=qthick equation so it is valid independantly of the state
        #as long as optically thick
        T_0=((3*kappa_0)*(1-b)*(1-p)/(4*sigma_boltzmann_cgs)*\
            func_H_R(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j,mode=mode)* \
            (c_cgs**4/(8*np.pi*G_cgs*m_BH_cgs))*(M_Edd*mdot_r)*r_arr**(-2)*\
            func_rho_0(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j,mode=mode)**(alpha_tau+1)\
                )**(1/(4-beta_tau))

        return T_0

    return T0_gen()

    # if mode=='rad':
    #     return T0_rad_dom()
    # elif mode=='gaz':
    #     return T0_gaz_dom()
    #
    # elif mode=='bridge':
    #     P_gaz_gaz = func_P_gaz(r_arr, mdot_in, kappa_0, alpha_tau, beta_tau, eta, mu, b, p, alpha_0, m_BH, r_j,
    #                            mode='gaz')
    #     P_rad_gaz=func_P_rad(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j,
    #                          mode='gaz')
    #
    #     P_rad_dom_mask=P_rad_gaz>P_gaz_gaz
    #
    #     T0_gaz=T0_gaz_dom()
    #     T0_rad=T0_rad_dom()
    #
    #     T0_bridge=np.array([T0_rad[i] if P_rad_dom_mask[i] else T0_gaz[i] for i in range(len(r_arr))])
    #
    #     return T0_bridge


def func_K_r(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta=0.1,mu=0.1,b=0,p=0,alpha_0=10,m_BH=1,r_j=6,mode='bridge'):

    '''
    wrapper for the K_r (opacity) solution for a standard SAD in the P_gaz dominant regime

    r: radius in Rg

    mdot_in:mass accretion rate at the starting radius

    kappa_0,alpha_tau,beta_tau: coefficients to compute the opacity (see above)

    mu: magnetization

    b: Jet Power index
    p: ejection index

    both default to 0

    p is constant along r, but if b is not constant the code should be edited

    alpha_0= viscosity coefficient, can also be seen as alpha_m*Prandlt_m, value used with mu to get the
                        viscosity parameter of the disk

    m_BH: Black Hole mass in solar masses
    '''

    test_1=func_rho_0(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j,mode=mode)
    test_2=func_T_0(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j,mode=mode)
    K_r=kappa_0*func_rho_0(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j,mode=mode)**(alpha_tau)*\
                func_T_0(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j,mode=mode)**(beta_tau)

    return K_r

def func_Tau(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta=0.1,mu=0.1,b=0,p=0,alpha_0=10,m_BH=1,r_j=6,mode='bridge'):

    '''
    wrapper for the Tau (optical depth) solution for a standard SAD in the P_gaz dominant regime

    r: radius in Rg

    mdot_in:mass accretion rate at the starting radius

    kappa_0,alpha_tau,beta_tau: coefficients to compute the opacity (see above)

    mu: magnetization

    b: Jet Power index
    p: ejection index

    both default to 0

    p is constant along r, but if b is not constant the code should be edited

    alpha_0= viscosity coefficient, can also be seen as alpha_m*Prandlt_m, value used with mu to get the
                        viscosity parameter of the disk

    m_BH: Black Hole mass in solar masses
    '''

    #defining useful mass dependant constants
    m_BH_cgs=m_BH*Msol_cgs
    Rg_cgs=G_cgs*m_BH_cgs/c_cgs**2

    #note: we fetch back H as H/R*R, with a conversion to cgs
    Tau=func_K_r(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j,mode=mode)*\
        func_rho_0(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j,mode=mode)*\
        func_H_R(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j,mode=mode)*r_arr*Rg_cgs

    return Tau
def func_P_gaz(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta=0.1,mu=0.1,b=0,p=0,alpha_0=10,m_BH=1,r_j=6,mode='bridge'):

    '''
    wrapper for the gas pressure solution for a standard SAD
    important: only in the P_gaz dominant regime

    r: radius in Rg

    mdot_in:mass accretion rate at the starting radius

    kappa_0,alpha_tau,beta_tau: coefficients to compute the opacity (see above)

    mu: magnetization

    b: Jet Power index
    p: ejection index

    both default to 0

    p is constant along r, but if b is not constant the code should be edited

    alpha_0= viscosity coefficient, can also be seen as alpha_m*Prandlt_m, value used with mu to get the
                        viscosity parameter of the disk

    m_BH: Black Hole mass in solar masses
    '''

    m_BH_cgs=m_BH*Msol_cgs
    Rg_cgs=G_cgs*m_BH_cgs/c_cgs**2

    n_star=1/(sigma_thomson_cgs*Rg_cgs)
    P_star=m_p_cgs * n_star * c_cgs ** 2

    L_Edd=1.26e38*m_BH

    #radial dependant mdot in CGS units
    mdot_r = (r_arr/r_j) ** (p) * mdot_in

    #computed with P_gaz=2*n_e*k*T0=2*rho_0/mp*k*T_0
    #P_gaz=2*func_rho_0(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j,mode=mode)/m_p_cgs*\
    #        k_boltzmann_cgs*func_T_0(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j,mode=mode)

    # #other way to compute it
    #P_gaz=func_rho_0(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j)*G_cgs*m_BH_cgs*\
    #         func_H_R(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j)**2*\
    #         1/(Rg_cgs*r_arr)

    P_gaz=P_star*mdot_r*r_arr**(-5/2)*1/(alpha_0*mu**(1/2)* \
           func_H_R(r_arr, mdot_in, kappa_0, alpha_tau, beta_tau, eta, mu, b, p, alpha_0, m_BH, r_j,mode=mode))

    return P_gaz

def func_P_rad(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta=0.1,mu=0.1,b=0,p=0,alpha_0=10,m_BH=1,r_j=6,mode='bridge'):

    '''
    wrapper for the radiation pressure solution for a standard SAD
    important: only in the P_gaz dominant regime

    r: radius in Rg

    mdot_in:mass accretion rate at the starting radius

    kappa_0,alpha_tau,beta_tau: coefficients to compute the opacity (see above)

    mu: magnetization

    b: Jet Power index
    p: ejection index

    both default to 0

    p is constant along r, but if b is not constant the code should be edited

    alpha_0= viscosity coefficient, can also be seen as alpha_m*Prandlt_m, value used with mu to get the
                        viscosity parameter of the disk

    m_BH: Black Hole mass in solar masses
    '''

    P_rad=(4*sigma_boltzmann_cgs/(3*c_cgs))*func_T_0(r_arr,mdot_in,kappa_0,alpha_tau,beta_tau,eta,mu,b,p,alpha_0,m_BH,r_j,
                                                     mode=mode)**4

    return P_rad

with st.sidebar.expander('Thermal structure computation'):
    compute_tstruct=st.checkbox('Compute thermal structure')

    struct_sol_mode=st.radio('Use solution from:',('Selected solution','Manual parameter input'),
                             index=1)
    st.text('Will only be considered\n if a single solution\n has been selected previously')

    P_dom_regime=st.radio('Regime to use',('bridge','gaz','rad'))

    st.title('Parameter inputs')

    struct_input_rj=st.number_input(r'Standard disk starting radius in Rg',value=6,format='%.3e')

    struct_input_mdot = st.number_input(r'mdot_in',value=1e0,format='%.3e')

    struct_input_eta = st.number_input(r'$\eta$ (radiative efficiency)', value=1e-1, format='%.3e')

    struct_input_mu = st.number_input(r'$\mu$ (magnetization)', value=1e-3, format='%.3e')

    struct_input_b = st.number_input(r'$b$ (jet power)', value=0, format='%.3e')

    struct_input_p = st.number_input(r'$p$ (ejection index)', value=0.1, format='%.3e')

    struct_input_alpha_0 = st.number_input(r'$\alpha_0$ (viscosity)', value=7, format='%.3e')
    #see salvesen16 or Jacquemin19

    struct_input_m_BH = st.number_input(r'$M_{BH}$ in $M_\odot$', value=8, format='%.3e')

if struct_sol_mode=='Manual parameter input':

    struct_rj = struct_input_rj
    struct_mdot = struct_input_mdot
    struct_eta = struct_input_eta
    struct_mu = struct_input_mu
    struct_b = struct_input_b
    struct_p = struct_input_p
    struct_alpha_0 = struct_input_alpha_0
    struct_m_BH = struct_input_m_BH

# elif struct_sol_mode is 'Selected solution':
#
#     #will need to be changed
#     struct_rj = struct_input_rj
#     struct_mdot = struct_input_mdot
#     struct_eta = struct_input_eta
#     struct_mu = struct_input_mu
#     struct_b = struct_input_b
#     struct_p = struct_input_p
#     struct_alpha_0 = struct_input_alpha_0
#     struct_m_BH = struct_input_m_BH

if compute_tstruct:

    #creating a range of radiuses from the inside to the outside
    struct_rsph = np.logspace(np.log10(struct_rj), 8, 5000)

    K_r_arr=np.zeros((n_regimes,len(struct_rsph)))

    #computing each individual opacity regime within this radius sample
    # for mode in 'gaz','rad'
    for id_K_r in range(n_regimes):
        K_r_arr[id_K_r]=func_K_r(struct_rsph,struct_mdot,
                                 kappa_0_arr[id_K_r],alpha_tau_arr[id_K_r],beta_tau_arr[id_K_r],
                                 struct_eta,struct_mu,struct_b,struct_p,struct_alpha_0,struct_m_BH,struct_rj,mode=P_dom_regime)

    max_opacity_mask=np.argmax(K_r_arr,0)

    #creating an array with the values of the dominant opacity regime at each radius
    kappa_0_dom_arr=np.array([kappa_0_arr[max_opacity_mask[i]] for i in range(len(struct_rsph))])
    alpha_tau_dom_arr = np.array([alpha_tau_arr[max_opacity_mask[i]] for i in range(len(struct_rsph))])
    beta_tau_dom_arr = np.array([beta_tau_arr[max_opacity_mask[i]] for i in range(len(struct_rsph))])

    #computing the main quantities
    struct_HR=func_H_R(struct_rsph,struct_mdot,kappa_0_dom_arr,alpha_tau_dom_arr,beta_tau_dom_arr,
                        struct_eta,struct_mu,struct_b,struct_p,struct_alpha_0,struct_m_BH,struct_rj,mode=P_dom_regime)

    struct_rho_0=func_rho_0(struct_rsph,struct_mdot,kappa_0_dom_arr,alpha_tau_dom_arr,beta_tau_dom_arr,
                        struct_eta,struct_mu,struct_b,struct_p,struct_alpha_0,struct_m_BH,struct_rj,mode=P_dom_regime)

    struct_T_0=func_T_0(struct_rsph,struct_mdot,kappa_0_dom_arr,alpha_tau_dom_arr,beta_tau_dom_arr,
                        struct_eta,struct_mu,struct_b,struct_p,struct_alpha_0,struct_m_BH,struct_rj,mode=P_dom_regime)

    struct_Tau=func_Tau(struct_rsph,struct_mdot,kappa_0_dom_arr,alpha_tau_dom_arr,beta_tau_dom_arr,
                        struct_eta,struct_mu,struct_b,struct_p,struct_alpha_0,struct_m_BH,struct_rj,mode=P_dom_regime)

    struct_P_gaz=func_P_gaz(struct_rsph,struct_mdot,kappa_0_dom_arr,alpha_tau_dom_arr,beta_tau_dom_arr,
                        struct_eta,struct_mu,struct_b,struct_p,struct_alpha_0,struct_m_BH,struct_rj,mode=P_dom_regime)

    struct_P_rad = func_P_rad(struct_rsph, struct_mdot, kappa_0_dom_arr, alpha_tau_dom_arr, beta_tau_dom_arr,
                            struct_eta, struct_mu, struct_b, struct_p, struct_alpha_0, struct_m_BH,struct_rj,mode=P_dom_regime)

    #figures
    fig_HR=plotly_line_wrapper(struct_rsph,struct_HR,log_x=True,log_y=True,xaxis_title='radius (Rg)', yaxis_title=r'H/R', legend='HR',
                                       name='H/R',showlegend=True,figwidth=515)

    fig_rho_0 = make_subplots(specs=[[{"secondary_y": True}]])
    fig_rho_0.update_layout(width=515)
    #giving a name to the secondary y axis
    fig_rho_0.update_yaxes(title_text="n_e (cgs)",gridcolor = 'rgba(0.5,0.5,.5,0.)',secondary_y=True)

    #adding second axis for the rho figure
    fig_rho_0.add_trace(go.Scatter(x=struct_rsph,y=struct_rho_0/m_p_cgs,
                                   line=dict(color='rgba(0.,0.,0.,0.)'),name='n_e'),
                                    secondary_y=True,)

    fig_rho_0=plotly_line_wrapper(struct_rsph,struct_rho_0,ex_fig=fig_rho_0,log_x=True,log_y=True,xaxis_title='radius (Rg)', yaxis_title=r' density (cgs)', legend='ρ_0',
                                       name='ρ_0',showlegend=True)



    fig_T_0=plotly_line_wrapper(struct_rsph,struct_T_0,log_x=True,log_y=True,xaxis_title='radius (Rg)', yaxis_title=r' Temperature (K)', legend='T_0',
                                       name='T_0',showlegend=True,figwidth=515)

    fig_Tau=plotly_line_wrapper(struct_rsph,struct_Tau,log_x=True,log_y=True,xaxis_title='radius (Rg)', yaxis_title=r'optical depth', legend='Tau',
                                       name='Tau',showlegend=True,figwidth=515)

    fig_P=plotly_line_wrapper(struct_rsph,struct_P_gaz,log_x=True,log_y=True,xaxis_title='radius (Rg)', yaxis_title=r' Pressure (cgs)', legend='P_gaz',line_color='blue',
                                       name='P_gaz',showlegend=True,figwidth=515)

    fig_P=plotly_line_wrapper(struct_rsph,struct_P_rad,ex_fig=fig_P,log_x=True,log_y=True,
                              xaxis_title='radius (Rg)', yaxis_title=r' Pressure (cgs)', legend='P_rad',line_color='yellow',
                                       name='P_rad',showlegend=True,figwidth=515)

    #plot the opacity regime evolutions
    fig_opacity=None
    for i in range(n_regimes):
        fig_opacity = plotly_line_wrapper(struct_rsph, K_r_arr[i],ex_fig=fig_opacity,
                                       log_x=True, log_y=True,line_color=col_regimes[i],
                                       xaxis_title='radius (Rg)', yaxis_title=r'opacity ($cm^2/g$)', legend=name_regimes[i],
                                       name=name_regimes[i],showlegend=True,figwidth=515)

    #computing the intervals where each non-validity condition applies
    rads_rad_dom=np.argwhere(struct_P_rad>struct_P_gaz).T[0]
    rads_non_thick=np.argwhere(struct_Tau<100).T[0]

    non_thick_regs=list(interval_extract(rads_non_thick))
    rad_dom_regs=list(interval_extract(rads_rad_dom))

    #adding the non-optically thick and Pressure dominated regions in each graph:
    for fig in [fig_HR,fig_rho_0,fig_T_0,fig_P,fig_opacity,fig_Tau]:
        #adding each zones to the graphs
        for i_reg, reg in enumerate(non_thick_regs):
            fig.add_vrect(x0=struct_rsph[reg[0]], x1=struct_rsph[reg[1]], line_width=0, fillcolor="grey", opacity=0.2,
                           name='Tau<100 regions',
                          label=dict(text='Tau<100 regions' if i_reg==0 else '',
                                    textposition="top center",
                                    font=dict(size=20, family="Times New Roman")))

        for i_reg, reg in enumerate(rad_dom_regs):
            fig.add_vrect(x0=struct_rsph[reg[0]], x1=struct_rsph[reg[1]], line_width=0, fillcolor="purple", opacity=0.2,
                           name='P_rad dominated region',
                          label=dict(text='P_rad dominated region' if i_reg==0 else '',
                                    textposition="top center",
                                    font=dict(size=20, family="Times New Roman")))



    with tab_tstruct:
        #placing them accordingly
        col_1,col_2,col_3=st.columns(3)
        with col_1:
            st.plotly_chart(fig_HR,use_container_width=False,theme=None)
            st.plotly_chart(fig_P, use_container_width=False, theme=None)

        with col_2:
            st.plotly_chart(fig_rho_0,use_container_width=False,theme=None)
            st.plotly_chart(fig_opacity, use_container_width=False, theme=None)

        with col_3:
            st.plotly_chart(fig_T_0,use_container_width=False,theme=None)
            st.plotly_chart(fig_Tau, use_container_width=False, theme=None)
