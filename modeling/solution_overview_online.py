import os,sys
import numpy as np
import streamlit as st
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

from solution_tools import func_density_sol,func_vel_sol,func_logxi_sol,func_nh_sol,load_solutions,sample_angle,interp_yaxis

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

else:

    update_online = False

    if not check_password():
        st.stop()

    solutions_path='/app/winds/modeling/solution_dumps/nathan_init.txt'
    possible_sol_path = '/app/winds/modeling/solution_dumps/super_a_0.0.dat'

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
        bgcolor='rgba(0.,0.,0.,0.)',font=dict(color="white")))

# fig_scatter.update_xaxes(tickfont=dict(color='white'),title_font_color="white")
# fig_scatter.update_yaxes(tickfont=dict(color='white'),title_font_color="white")

tab_p_mu, tab_sol, tab_sol_radial,tab_explo= st.tabs(["Solution selection", "Angular distribution", "radial distribution","parameter exploration"])

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
    st.stop()

if selected_points is not None:

    #only selected the points from the second trace with the curvenumber test
    selected_sol_p_mu=np.array([[selected_points[i]['y'],selected_points[i]['x']] for i in range(len(selected_points)) if selected_points[i]['curveNumber']==1])
    n_sel=len(selected_sol_p_mu)

    if n_sel==0:
        st.warning('None of the selected points have a MHD solution.')
        st.stop()

#fetching the individual solutions with these parameters
selected_sol_mask=np.array([np.all(np.array([elem[0][2:4] for elem in sol_splitsol])==selected_sol_p_mu[i],axis=1)\
                            for i in range(n_sel)]).any(0)

selected_mhd_sol=sol_splitsol[selected_sol_mask]

# ordering the mask in order or reading (increasing mu)

sol_order=selected_sol_p_mu.T[1].argsort() if sol_order_select =='Increasing mu' else\
            selected_sol_p_mu.T[0].argsort() if sol_order_select == 'Increasing p' else True

selected_mhd_sol=selected_mhd_sol[sol_order]
selected_sol_p_mu=selected_sol_p_mu[sol_order]


#doing this without direct transpositions because the arrays arent regular due to uneven angle sampling

sol_z_over_r=np.array([selected_mhd_sol[i].T[7] for i in range(n_sel)],dtype=object)
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
                        line_color='lightskyblue',ex_fig=None,name='',showlegend=False,split_ls_x=None,linedash='solid'):

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
        fig_line=go.Figure()

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

def angle_plot(x_arr,y_arr,log_x=False,log_y='auto',xaxis_title='',yaxis_title='',legend=False,sampl_angles_arr=None,compt_angles_arr=None,cmap='cividis',legend_lines=True,legend_points=True):

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

        '''
        adding a split depending on whether the line is compton thick or not to allow distinguishing the compton
         thick zones for several solutions
        '''

        if n_sel>1:

            #computing the position of the compton angle on the y axis
            compt_y=interp_yaxis(compt_angle,x,y)

            fig_line=plotly_line_wrapper(x,y,
                                         log_x=log_x,log_y='auto',
                                         xaxis_title=xaxis_title,yaxis_title=yaxis_title,legend=legend,ex_fig=fig_line,
                                         name=line_name,line_color=elem_color,showlegend=legend_lines,split_ls_x=compt_angle,linedash=['solid','dash'])

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



if split_angle:

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
                            xaxis_title =r'$\theta \; (°)$', yaxis_title = r'$\rho_{mhd}$',
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
    st.stop()

if n_sel==1 and not split_angle:
    with tab_sol_radial:
        st.info('Activate angle sampling in the sidebar to see radial distributions.')
    with tab_explo:
        st.info('Activate angle sampling in the sidebar to see parameter exploration.')
    st.stop()

def radial_plot(rad,sol_sampl,angl_sampl,log_x=False,log_y=False,xaxis_title='',yaxis_title='',legend=False,
                cmap='plasma_r',logxi_ids=None,yrange=None):

    norm_angl= (angl_sampl-val_angle_low)/(val_angle_high-val_angle_low)

    ang_colors = sample_colorscale(cmap,norm_angl)

    #creating the theme with the first line
    fig_rad=plotly_line_wrapper(rad[0],sol_sampl[0],log_x=log_x,log_y='auto',xaxis_title=xaxis_title,yaxis_title=yaxis_title,line_color=ang_colors[0],legend=True)

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
            hovertemplate="<b>" + 'theta='+str(elem_angl) + "</b><br>" +
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

with tab_sol_radial:

    n_sol=len(selected_sol_split_angle[0])
    
    sol_sampl_z_over_r = selected_sol_split_angle[0].T[7]
    sol_sampl_angle = selected_sol_split_angle[0].T[8]

    sol_sampl_r_cyl_r0 = selected_sol_split_angle[0].T[9]
    sol_sampl_rho_mhd = selected_sol_split_angle[0].T[10]

    sol_sampl_ur, sol_sampl_uphi, sol_sampl_uz = selected_sol_split_angle[0].T[11:14]
    sol_sampl_br, sol_sampl_bphi, sol_sampl_bz = selected_sol_split_angle[0].T[15:18]

    sol_p_mhd=selected_sol_split_angle[0][0][3]

    mdot_mhd=mdot_obs*12

    cyl_cst_sampl=np.sqrt(1+sol_sampl_z_over_r**2)

    r_sph_sampl=np.array([np.logspace(np.log10(rj*cyl_cst_sampl[i]),7,300) for i in range(n_sol)])

    n_sampl=np.array([func_density_sol(r_sph_sampl[i],sol_sampl_z_over_r[i],sol_sampl_rho_mhd[i],sol_p_mhd,
                                       mdot_mhd,m_BH) for i in range(n_sol)])

    logxi_sampl=np.array([func_logxi_sol(r_sph_sampl[i],sol_sampl_z_over_r[i],val_L_source,sol_sampl_rho_mhd[i],
                                         sol_p_mhd,mdot_mhd,m_BH) for i in range(n_sol)])

    nh_sampl=np.array([func_nh_sol(r_sph_sampl[i],rj*cyl_cst_sampl[i],sol_sampl_z_over_r[i],
                                   sol_sampl_rho_mhd[i],sol_p_mhd,mdot_mhd,m_BH) for i in range(n_sol)])

    #here all speeds are divided by 1e5 to go to km/s
    ur_sampl=np.array([func_vel_sol('r',r_sph_sampl[i],sol_sampl_z_over_r[i],sol_sampl_ur[i],
                                    sol_sampl_uphi[i],sol_sampl_uz[i]) for i in range(n_sol)])/1e5

    uphi_sampl=np.array([func_vel_sol('phi',r_sph_sampl[i],sol_sampl_z_over_r[i],sol_sampl_ur[i],
                                      sol_sampl_uphi[i],sol_sampl_uz[i]) for i in range(n_sol)])/1e5

    uz_sampl=np.array([func_vel_sol('z',r_sph_sampl[i],sol_sampl_z_over_r[i],sol_sampl_ur[i],
                                    sol_sampl_uphi[i],sol_sampl_uz[i]) for i in range(n_sol)])/1e5

    uobs_sampl=np.array([func_vel_sol('obs',r_sph_sampl[i],sol_sampl_z_over_r[i],sol_sampl_ur[i],
                                      sol_sampl_uphi[i],sol_sampl_uz[i]) for i in range(n_sol)])/1e5

    #fetching the positions at which logxi=6 for each angle
    logxi_6_ids=np.array([np.argmin(abs(elem-6)) for elem in logxi_sampl])

    r_sph_nonthick_sampl=np.array([r_sph_sampl[i][logxi_6_ids[i]:] for i in range(n_sol)],dtype=object)
    
    nh_nonthick_sampl=np.array([func_nh_sol(r_sph_sampl[i][logxi_6_ids[i]:],r_sph_sampl[i][logxi_6_ids[i]],
                        sol_sampl_z_over_r[i],sol_sampl_rho_mhd[i],sol_p_mhd,mdot_mhd,m_BH) for i in range(n_sol)],
                               dtype=object)

    col_a,col_b,col_c=st.columns(3)

    with col_a:
        radial_plot(r_sph_sampl,n_sampl,sol_sampl_angle,log_x=True,
                                    log_y=True,xaxis_title=r'$R_{sph}\;$ (Rg)',yaxis_title=r'$n\textrm{ (cgs)}$',
                    logxi_ids=logxi_6_ids)

        radial_plot(r_sph_sampl,ur_sampl,sol_sampl_angle,log_x=True,
                                    log_y=True,xaxis_title=r'$R_{sph}\;$ (Rg)',yaxis_title=r'$v_{r}\textrm{ (km/s)}$',
                    logxi_ids=logxi_6_ids)
        
        radial_plot(r_sph_sampl,nh_sampl,sol_sampl_angle,log_x=True,log_y=True,
                    xaxis_title=r'$R_{sph}\;$ (Rg)',yaxis_title=r'$n_{h}\textrm{ (cm}^{-2}\textrm{)}$',
                    yrange=[np.log10(1e20),np.log10(1.5e24)])

    with col_b:
        radial_plot(r_sph_sampl,logxi_sampl, sol_sampl_angle, log_x=True,
                    log_y=False, xaxis_title=r'$R_{sph}\;$ (Rg)', yaxis_title=r'$log\xi$',
                    logxi_ids=logxi_6_ids)

        radial_plot(r_sph_sampl, uphi_sampl, sol_sampl_angle, log_x=True,
                    log_y=True, xaxis_title=r'$R_{sph}\;$ (Rg)', yaxis_title=r'$v_{\phi}\textrm{ (km/s)}$',
                    logxi_ids=logxi_6_ids)

        radial_plot(r_sph_nonthick_sampl,nh_nonthick_sampl,sol_sampl_angle,log_x=True,log_y=True,
                    xaxis_title=r'$R_{sph}\;$ (Rg)',yaxis_title=r'$n_{h}^{\textrm{log}\xi\leq6}\textrm{(cm}^{-2}\textrm{)}$',
                    yrange=[np.log10(1e20),np.log10(1.5e24)])

    with col_c:

        radial_plot(r_sph_sampl, uobs_sampl, sol_sampl_angle, log_x=True,
                    log_y=True, xaxis_title=r'$R_{sph}\;$ (Rg)', yaxis_title=r'$v_{obs}\textrm{ (km/s)}$',
                    logxi_ids=logxi_6_ids)

        radial_plot(r_sph_sampl, uz_sampl, sol_sampl_angle, log_x=True,
                    log_y=True, xaxis_title=r'$R_{sph}\;$ (Rg)', yaxis_title=r'$v_{z}\textrm{ (km/s)}$',
                    logxi_ids=logxi_6_ids)



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
                    log_y=True, xaxis_title=r'$R_{sph}\;$ (Rg)', yaxis_title=r'L/L$_{Edd}$')