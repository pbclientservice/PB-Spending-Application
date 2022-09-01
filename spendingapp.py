
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px


# Streamlit - allows ability to refresh app to see code changes
st.cache(allow_output_mutation=True)



st.title('Endowment Spending Simulator')
st.write('Important: Before proceeding below, complete the required inputs in the left hand column')
st.write('Yale Model = [(80%)(Prior QTR Distribution)(1 + QTR CPI)] + [(20%)(Rolling Quarter Average MV)(QTR Spend Rate)]')
# Questions to gather input
st.sidebar.markdown('## Enter Below Variables First')
cpi = st.sidebar.text_input("Enter an estimate for inflation (ex - difference between real and nominal expected return)")
cpi_std = st.sidebar.text_input("Enter an estimate for inflation volatility (Default = 2%)")
t_intervals = st.sidebar.text_input("What is the simulation period? Enter in quarters (ex - 10 years = 40)")
options = st.sidebar.multiselect(
        'Select percentiles for comparison',
        [.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]
        )
rolling_quarters = st.sidebar.text_input("How many rolling quarters are used in the spending calculation methodlogy?")
uploaded_file = st.sidebar.file_uploader('Drop in Excel with historical quarterly market values. The number of historical market values provided should equal the ROLLING period.')
if uploaded_file is not None:
    historic_values = pd.read_excel(uploaded_file)
    st.sidebar.dataframe(historic_values)
    if len(historic_values) != int(rolling_quarters):
        st.sidebar.write(':x: Error. Check length of provided market values.')
    else:
        st.sidebar.write(':white_check_mark: Confirmed length of provided market values')
uploaded_file_2 = st.sidebar.file_uploader('Drop in Excel if any fixed activity. Enter withdrawals as a negative (-) and contributions as a positive (+) number.')
if uploaded_file_2 is not None:
    fixed_activity_upload = pd.read_excel(uploaded_file_2)
    t_intervals_slice = int(t_intervals)
    fixed_spending_activity = pd.DataFrame(fixed_activity_upload.iloc[0:t_intervals_slice,1]).fillna(0)
    other_fixed_activity = pd.DataFrame(fixed_activity_upload.iloc[0:t_intervals_slice,2]).fillna(0)
    st.sidebar.write('Fixed Spending by Period (values entered should typically be negative, otherwise will reduce / serve as an offset to dollars spent):')
    st.sidebar.dataframe(fixed_spending_activity)
    st.sidebar.write('Other Fixed Activity by Period (negative values for outflows and positive values for inflows):')
    st.sidebar.dataframe(other_fixed_activity)

st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')


sim = []

# Functions to gather input for variables in each simulation.

def annual_return():
    annual_ret = st.text_input(f"Nominal arithmetic annualized expected return for Sim {sim}")
    return annual_ret

def annual_std():
    annual_std = st.text_input(f"Standard deviation for Sim {sim}")
    return annual_std

def annual_spending():
    annual_spending = st.text_input(f"Annual spend rate for Sim {sim}")
    return annual_spending

def annual_spending_initial():
    annual_spending_initial = st.text_input(f"First annual spend rate for Sim {sim}?")
    return annual_spending_initial

def annual_spending_initial_duration():
    annual_spending_initial_duration = st.text_input(f'First spend rate duration (quarters) for Sim {sim}?')
    return annual_spending_initial_duration

def annual_spending_final():
    annual_spending_final = st.text_input(f"Long-term annual spend rate for Sim {sim}?")
    return annual_spending_final

# Function to compute results assuming one constant spending rate

def compute_constant(annual_ret, annual_std, annual_spending, rolling_quarters, cpi, cpi_std, t_intervals, yale_input, prior_qtr_spend):
            
    annual_ret = float(annual_ret) / 100
    annual_std = float(annual_std) / 100
    annual_spending = float(annual_spending) / 100
    rolling_quarters = int(rolling_quarters)
    cpi = float(cpi) / 100
    cpi_std = float(cpi_std) / 100
    t_intervals = int(t_intervals) + rolling_quarters
    
    quarterly_ret = annual_ret/4
    quarterly_spending = annual_spending/4
    quarterly_stdev = annual_std / (4**0.5)
    quarterly_cpi_std = cpi_std / (4**0.5)
    quarterly_returns = 1 + np.random.normal(quarterly_ret, quarterly_stdev, (t_intervals,10001))
    spend = np.zeros_like(quarterly_returns)
    if yale_input == 1:
        spend[rolling_quarters-1] = prior_qtr_spend 

    portfolio = np.zeros_like(quarterly_returns)
    portfolio[0:rolling_quarters]= historic_values[0:rolling_quarters]
    quarter_cpi = cpi / 4
    quarter_cpi = 1 + np.random.normal(quarter_cpi, quarterly_cpi_std, (t_intervals,10001))
    portfolio_real = np.zeros_like(quarterly_returns)
    portfolio_real[0:rolling_quarters] = historic_values[0:rolling_quarters]
    spend_real = np.zeros_like(quarterly_returns)
    spend_real[0:rolling_quarters] = 0
    time_zero = rolling_quarters - 1
    inflation_discounter = np.zeros_like(quarterly_returns)
    inflation_discounter[0:rolling_quarters] = 1
    # Fixed Activity
    other_fixed_activity_sim = np.zeros_like(quarterly_returns)
    other_fixed_activity_sim[0:rolling_quarters] = 0
    fixed_spending_activity_sim = np.zeros_like(quarterly_returns)
    fixed_spending_activity_sim[0:rolling_quarters] = 0
    if uploaded_file_2 is not None:
        other_fixed_activity_sim[rolling_quarters:] = other_fixed_activity[0:]
        fixed_spending_activity_sim[rolling_quarters:] = fixed_spending_activity[0:]
    else:
        other_fixed_activity_sim[rolling_quarters:] = 0
        fixed_spending_activity_sim[rolling_quarters:] = 0
        


    if yale_input == 0:
        
        for t in range (rolling_quarters, t_intervals):
            IC_mv = pd.DataFrame(portfolio)
            IC_rolling_mv = IC_mv.rolling(rolling_quarters, min_periods=1).mean()
            IC_rolling_mv = np.array(IC_rolling_mv)
            spend[t] = (quarterly_spending*IC_rolling_mv[t-1])-fixed_spending_activity_sim[t]
            portfolio[t] = (portfolio[t-1]*quarterly_returns[t])-spend[t]+other_fixed_activity_sim[t]
            inflation_discounter[t] = inflation_discounter[t-1] * quarter_cpi[t]
            portfolio_real[t] = (portfolio[t] / inflation_discounter[t])
            spend_real[t] = (spend[t] / inflation_discounter[t])

    elif yale_input == 1:

        for t in range (rolling_quarters, t_intervals):
            IC_mv = pd.DataFrame(portfolio)
            IC_rolling_mv = IC_mv.rolling(rolling_quarters, min_periods=1).mean()
            IC_rolling_mv = np.array(IC_rolling_mv)
            spend[t] = (spend[t-1]*0.8*(quarter_cpi[t]))+(.2*(quarterly_spending*IC_rolling_mv[t-1]))-fixed_spending_activity_sim[t]
            portfolio[t] = (portfolio[t-1]*quarterly_returns[t])-spend[t]+other_fixed_activity_sim[t]
            inflation_discounter[t] = inflation_discounter[t-1] * quarter_cpi[t]
            portfolio_real[t] = (portfolio[t] / inflation_discounter[t])
            spend_real[t] = (spend[t] / inflation_discounter[t])


    portfolio_real_df =  pd.DataFrame(portfolio_real[time_zero:]).reset_index(drop=True)
    spend_real_df = pd.DataFrame(spend_real[rolling_quarters:]).reset_index(drop=True)
    spend_real_df.index = np.arange(1, len(spend_real_df)+1)

    portfolio_nom_df =  pd.DataFrame(portfolio[time_zero:]).reset_index(drop=True)
    spend_nom_df = pd.DataFrame(spend[rolling_quarters:]).reset_index(drop=True)
    spend_nom_df.index = np.arange(1, len(spend_nom_df)+1)


    percentiles_real = portfolio_real_df.quantile(options, axis = 1)
    percentiles_real = pd.DataFrame.transpose(percentiles_real)
    percentiles_real_spend = spend_real_df.quantile(options, axis = 1) 
    percentiles_real_spend = pd.DataFrame.transpose(percentiles_real_spend)



    percentiles_nominal = portfolio_nom_df.quantile(options, axis = 1)
    percentiles_nominal = pd.DataFrame.transpose(percentiles_nominal)
    percentiles_nom_spend = spend_nom_df.quantile(options, axis = 1) 
    percentiles_nom_spend = pd.DataFrame.transpose(percentiles_nom_spend)


    return portfolio_real, spend_real_df, portfolio, spend_nom_df, percentiles_real, percentiles_real_spend, percentiles_nominal, percentiles_nom_spend


# Function to compute results assuming a temporary short-term and long-term spending rate

def compute_variable(annual_ret, annual_std, annual_spending_initial, annual_spending_initial_duration, annual_spending_final, rolling_quarters, cpi, cpi_std, t_intervals):

    annual_ret = float(annual_ret) / 100
    annual_std = float(annual_std) / 100
    annual_spending_initial = float(annual_spending_initial) / 100
    annual_spending_final = float(annual_spending_final) / 100
    rolling_quarters = int(rolling_quarters)
    annual_spending_initial_duration = int(annual_spending_initial_duration) + rolling_quarters
    cpi = float(cpi) / 100
    cpi_std = float(cpi_std) / 100
    t_intervals = int(t_intervals) + rolling_quarters

    quarterly_ret = annual_ret/4
    quarterly_stdev = annual_std / (4**0.5)
    quarterly_cpi_std = cpi_std / (4**0.5)
    quarterly_spending_initial = annual_spending_initial/4
    quarterly_spending_final = annual_spending_final/4
    quarterly_returns = 1 + np.random.normal(quarterly_ret, quarterly_stdev, (t_intervals,10001))
    spend = np.zeros_like(quarterly_returns)
    portfolio = np.zeros_like(quarterly_returns)
    portfolio[0:rolling_quarters]= historic_values[0:rolling_quarters]
    quarter_cpi = cpi / 4
    quarter_cpi = 1 + np.random.normal(quarter_cpi, quarterly_cpi_std, (t_intervals,10001))
    portfolio_real = np.zeros_like(quarterly_returns)
    portfolio_real[0:rolling_quarters] = historic_values[0:rolling_quarters]
    spend_real = np.zeros_like(quarterly_returns)
    spend_real[0:rolling_quarters] = 0
    time_zero = rolling_quarters - 1
    inflation_discounter = np.zeros_like(quarterly_returns)
    inflation_discounter[0:rolling_quarters] = 1
    # Fixed Activity
    other_fixed_activity_sim = np.zeros_like(quarterly_returns)
    other_fixed_activity_sim[0:rolling_quarters] = 0
    fixed_spending_activity_sim = np.zeros_like(quarterly_returns)
    fixed_spending_activity_sim[0:rolling_quarters] = 0
    if uploaded_file_2 is not None:
        other_fixed_activity_sim[rolling_quarters:] = other_fixed_activity[0:]
        fixed_spending_activity_sim[rolling_quarters:] = fixed_spending_activity[0:]
    else:
        other_fixed_activity_sim[rolling_quarters:] = 0
        fixed_spending_activity_sim[rolling_quarters:] = 0
    
    #simulation
    for t in range (rolling_quarters, t_intervals):
        IC_mv = pd.DataFrame(portfolio)
        IC_rolling_mv = IC_mv.rolling(rolling_quarters, min_periods=1).mean()
        IC_rolling_mv = np.array(IC_rolling_mv)
        if t <= annual_spending_initial_duration:
            quarterly_spending = quarterly_spending_initial
        else: 
            quarterly_spending = quarterly_spending_final
        spend[t] = (quarterly_spending*IC_rolling_mv[t-1])-fixed_spending_activity_sim[t]
        portfolio[t] = (portfolio[t-1]*quarterly_returns[t])-spend[t]+other_fixed_activity_sim[t]
        inflation_discounter[t] = inflation_discounter[t-1] * quarter_cpi[t]
        portfolio_real[t] = (portfolio[t] / inflation_discounter[t])
        spend_real[t] = (spend[t] / inflation_discounter[t])

    portfolio_real_df =  pd.DataFrame(portfolio_real[time_zero:]).reset_index(drop=True)
    spend_real_df = pd.DataFrame(spend_real[rolling_quarters:]).reset_index(drop=True)
    spend_real_df.index = np.arange(1, len(spend_real_df)+1)

    portfolio_nom_df =  pd.DataFrame(portfolio[time_zero:]).reset_index(drop=True)
    spend_nom_df = pd.DataFrame(spend[rolling_quarters:]).reset_index(drop=True)
    spend_nom_df.index = np.arange(1, len(spend_nom_df)+1)


    percentiles_real = portfolio_real_df.quantile(options, axis = 1)
    percentiles_real = pd.DataFrame.transpose(percentiles_real)
    percentiles_real_spend = spend_real_df.quantile(options, axis = 1) 
    percentiles_real_spend = pd.DataFrame.transpose(percentiles_real_spend)


    percentiles_nominal = portfolio_nom_df.quantile(options, axis = 1)
    percentiles_nominal = pd.DataFrame.transpose(percentiles_nominal)
    percentiles_nom_spend = spend_nom_df.quantile(options, axis = 1) 
    percentiles_nom_spend = pd.DataFrame.transpose(percentiles_nom_spend)

    return portfolio_real, spend_real_df, portfolio, spend_nom_df, percentiles_real, percentiles_real_spend, percentiles_nominal, percentiles_nom_spend


# Streamlit browser display in columns

col1, col2, col3 = st.columns(3)
with col1:
    st.header("Sim 1")
    sim = 1
    spending_plan_1 = st.selectbox(
        '''Select a spending 
        plan for Sim 1''',
        ('', 'Constant (single spend rate)', 'Variable (multiple spend rates)')
    )
    if spending_plan_1 == 'Constant (single spend rate)':
        yale_1 = st.checkbox('Yale model for Sim 1?')
        if yale_1:
            yale_input_1 = 1
            prior_qtr_spend_1 = st.text_input('Input prior quarter spending amount in dollars for Sim 1')
            if prior_qtr_spend_1:
                prior_qtr_spend_1 = float(prior_qtr_spend_1)
            else:
                st.write(':octagonal_sign: Enter a value above to continue.')
        else:
            yale_input_1 = 0
            prior_qtr_spend_1 = 0
        annual_return_1 = annual_return()
        if annual_return_1 is not '':
            st.write(f'(Annual real return of {float(annual_return_1) - float(cpi)})')
        annual_std_1 = annual_std()
        annual_spending_1 = annual_spending()
        
    elif spending_plan_1 == 'Variable (multiple spend rates)':
        annual_return_1 = annual_return()
        if annual_return_1 is not '':
            st.write(f'(Annual real return of {float(annual_return_1) - float(cpi)})')
        annual_std_1 = annual_std()
        annual_spending_initial_1 = annual_spending_initial()
        annual_spending_initial_duration_1 = annual_spending_initial_duration()
        annual_spending_final_1 = annual_spending_final()
        
with col2:
    st.header("Sim 2")
    sim = 2
    spending_plan_2 = st.selectbox(
        '''Select a spending 
        plan for Sim 2''',
        ('', 'Constant (single spend rate)', 'Variable (multiple spend rates)')
    )
    if spending_plan_2 == 'Constant (single spend rate)':
        yale_2 = st.checkbox('Yale model for Sim 2?')
        if yale_2:
            yale_input_2 = 1
            prior_qtr_spend_2 = st.text_input('Input prior quarter spending amount in dollars for Sim 2')
            if prior_qtr_spend_2:
                prior_qtr_spend_2 = float(prior_qtr_spend_2)
            else:
                st.write(':octagonal_sign: Enter a value above to continue.')
        else:
            yale_input_2 = 0
            prior_qtr_spend_2 = 0
        annual_return_2 = annual_return()
        if annual_return_2 is not '':
            st.write(f'(Annual real return of {float(annual_return_2) - float(cpi)})')
        annual_std_2 = annual_std()
        annual_spending_2 = annual_spending()
    elif spending_plan_2 == 'Variable (multiple spend rates)':
        annual_return_2 = annual_return()
        if annual_return_2 is not '':
            st.write(f'(Annual real return of {float(annual_return_2) - float(cpi)})')
        annual_std_2 = annual_std()
        annual_spending_initial_2 = annual_spending_initial()
        annual_spending_initial_duration_2 = annual_spending_initial_duration()
        annual_spending_final_2 = annual_spending_final()

with col3:
    st.header("Sim 3")
    sim = 3
    spending_plan_3 = st.selectbox(
        '''Select a spending 
        plan for Sim 3''',
        ('', 'Constant (single spend rate)', 'Variable (multiple spend rates)')
    )
    if spending_plan_3 == 'Constant (single spend rate)':
        yale_3 = st.checkbox('Yale model for Sim 3?')
        if yale_3:
            yale_input_3 = 1
            prior_qtr_spend_3 = st.text_input('Input prior quarter spending amount in dollars for Sim 3')
            if prior_qtr_spend_3:
                prior_qtr_spend_3 = float(prior_qtr_spend_3)
            else:
                st.write(':octagonal_sign: Enter a value above to continue.')
        else:
            yale_input_3 = 0
            prior_qtr_spend_3 = 0
        annual_return_3 = annual_return()
        if annual_return_3 is not '':
            st.write(f'(Annual real return of {float(annual_return_3) - float(cpi)})')
        annual_std_3 = annual_std()
        annual_spending_3 = annual_spending()
    elif spending_plan_3 == 'Variable (multiple spend rates)':
        annual_return_3 = annual_return()
        if annual_return_3 is not '':
            st.write(f'(Annual real return of {float(annual_return_3) - float(cpi)})')
        annual_std_3 = annual_std()
        annual_spending_initial_3 = annual_spending_initial()
        annual_spending_initial_duration_3 = annual_spending_initial_duration()
        annual_spending_final_3 = annual_spending_final()

# Select box for real or nominal terms
st.write('Select to display visuals in Nominal terms. Unselect for Real terms.')
nom_check = st.checkbox('Nominal Terms')
st.write('(CSV output will include both real and nominal regardless of selection.)')



# Computes the user input results if real terms is preferred
# Returns several dataframe/list objects to be used in charting below

if not nom_check and st.button('Compute'):
    if spending_plan_1 == 'Constant (single spend rate)':
        portfolio_real_1, spend_real_df_1, portfolio_nom_1, spend_nom_df_1, percentiles_real_1, percentiles_real_spend_1, percentiles_nominal_1, percentiles_nom_spend_1 = compute_constant(annual_return_1, annual_std_1, annual_spending_1, rolling_quarters, cpi, cpi_std, t_intervals, yale_input_1, prior_qtr_spend_1)
        input_data_1 = pd.DataFrame({'Sim1 Inputs': [annual_return_1,annual_std_1,annual_spending_1,'NA','NA','NA',rolling_quarters,cpi,t_intervals, yale_input_1]},
            index = ['Annual Return', 'Annual Std.', 'Annual Spend Rate (if Constant)', 'Initial Spend Rate (if Variable)','Initial Spend Rate Duration (if Variable)','Long-Term Spend Rate (if Variable)','Rolling Period (Qtrs)', 'CPI', 'Simulation Period','Yale Model (1=Yes)']
            )

    elif spending_plan_1 == 'Variable (multiple spend rates)':
        portfolio_real_1, spend_real_df_1, portfolio_nom_1, spend_nom_df_1, percentiles_real_1, percentiles_real_spend_1, percentiles_nominal_1, percentiles_nom_spend_1 = compute_variable(annual_return_1, annual_std_1, annual_spending_initial_1, annual_spending_initial_duration_1, annual_spending_final_1, rolling_quarters, cpi, cpi_std, t_intervals)
        input_data_1 = pd.DataFrame({'Sim1 Inputs': [annual_return_1,annual_std_1,'NA', annual_spending_initial_1,annual_spending_initial_duration_1,annual_spending_final_1,rolling_quarters,cpi,t_intervals, 'NA']},
            index = ['Annual Return', 'Annual Std.', 'Annual Spend Rate (if Constant)', 'Initial Spend Rate (if Variable)','Initial Spend Rate Duration (if Variable)','Long-Term Spend Rate (if Variable)','Rolling Period (Qtrs)', 'CPI', 'Simulation Period','Yale Model (1=Yes)']
            )

    if spending_plan_2 == 'Constant (single spend rate)':
        portfolio_real_2, spend_real_df_2, portfolio_nom_2, spend_nom_df_2, percentiles_real_2, percentiles_real_spend_2, percentiles_nominal_2, percentiles_nom_spend_2 = compute_constant(annual_return_2, annual_std_2, annual_spending_2, rolling_quarters, cpi, cpi_std, t_intervals, yale_input_2, prior_qtr_spend_2)
        input_data_2 = pd.DataFrame({'Sim2 Inputs': [annual_return_2,annual_std_2,annual_spending_2,'NA','NA','NA',rolling_quarters,cpi,t_intervals, yale_input_2]},
            index = ['Annual Return', 'Annual Std.', 'Annual Spend Rate (if Constant)', 'Initial Spend Rate (if Variable)','Initial Spend Rate Duration (if Variable)','Long-Term Spend Rate (if Variable)','Rolling Period (Qtrs)', 'CPI', 'Simulation Period','Yale Model (1=Yes)']
            )

    elif spending_plan_2 == 'Variable (multiple spend rates)':
        portfolio_real_2, spend_real_df_2, portfolio_nom_2, spend_nom_df_2, percentiles_real_2, percentiles_real_spend_2, percentiles_nominal_2, percentiles_nom_spend_2 = compute_variable(annual_return_2, annual_std_2, annual_spending_initial_2, annual_spending_initial_duration_2, annual_spending_final_2, rolling_quarters, cpi, cpi_std, t_intervals)
        input_data_2 = pd.DataFrame({'Sim2 Inputs': [annual_return_2,annual_std_2,'NA', annual_spending_initial_2,annual_spending_initial_duration_2,annual_spending_final_2,rolling_quarters,cpi,t_intervals, 'NA']},
            index = ['Annual Return', 'Annual Std.', 'Annual Spend Rate (if Constant)', 'Initial Spend Rate (if Variable)','Initial Spend Rate Duration (if Variable)','Long-Term Spend Rate (if Variable)','Rolling Period (Qtrs)', 'CPI', 'Simulation Period','Yale Model (1=Yes)']
            )    

    if spending_plan_3 == 'Constant (single spend rate)':
        portfolio_real_3, spend_real_df_3, portfolio_nom_3, spend_nom_df_3, percentiles_real_3, percentiles_real_spend_3, percentiles_nominal_3, percentiles_nom_spend_3 = compute_constant(annual_return_3, annual_std_3, annual_spending_3, rolling_quarters, cpi, cpi_std, t_intervals, yale_input_3, prior_qtr_spend_3)
        input_data_3 = pd.DataFrame({'Sim3 Inputs': [annual_return_3,annual_std_3,annual_spending_3,'NA','NA','NA',rolling_quarters,cpi,t_intervals, yale_input_3]},
            index = ['Annual Return', 'Annual Std.', 'Annual Spend Rate (if Constant)', 'Initial Spend Rate (if Variable)','Initial Spend Rate Duration (if Variable)','Long-Term Spend Rate (if Variable)','Rolling Period (Qtrs)', 'CPI', 'Simulation Period','Yale Model (1=Yes)']
            )

    elif spending_plan_3 == 'Variable (multiple spend rates)':
        portfolio_real_3, spend_real_df_3, portfolio_nom_3, spend_nom_df_3, percentiles_real_3, percentiles_real_spend_3, percentiles_nominal_3, percentiles_nom_spend_3 = compute_variable(annual_return_3, annual_std_3, annual_spending_initial_3, annual_spending_initial_duration_3, annual_spending_final_3, rolling_quarters, cpi, cpi_std, t_intervals)
        input_data_3 = pd.DataFrame({'Sim3 Inputs': [annual_return_3,annual_std_3,'NA', annual_spending_initial_3,annual_spending_initial_duration_3,annual_spending_final_3,rolling_quarters,cpi,t_intervals, 'NA']},
            index = ['Annual Return', 'Annual Std.', 'Annual Spend Rate (if Constant)', 'Initial Spend Rate (if Variable)','Initial Spend Rate Duration (if Variable)','Long-Term Spend Rate (if Variable)','Rolling Period (Qtrs)', 'CPI', 'Simulation Period','Yale Model (1=Yes)']
            )


    # Takes the real portfolio market value results and organizes the output into a dataframe for the final ending market value only
    # if statements to allow flexibility if only 1 or 2 simulations conducted

    s1_mv = pd.DataFrame(portfolio_real_1[-1], columns=['Sim1'])
    if spending_plan_2 is not '':
        s2_mv = pd.DataFrame(portfolio_real_2[-1], columns=['Sim2'])
    if spending_plan_3 is not '':
        s3_mv = pd.DataFrame(portfolio_real_3[-1], columns=['Sim3'])

    # Combines each simulation into 1 dataframe
    if spending_plan_2 is '':
        s_combined_mv = s1_mv
        input_data = input_data_1
    elif spending_plan_3 is '':
        s_combined_mv = pd.concat([s1_mv,s2_mv], axis=1)
        input_data = pd.concat([input_data_1, input_data_2], axis=1)
    else:
        s_combined_mv = pd.concat([s1_mv,s2_mv,s3_mv], axis=1)
        input_data = pd.concat([input_data_1, input_data_2, input_data_3], axis=1)


    # Repeats the above process for the $ spending data
    # Code is slightly different than above given the output of the spending data being in a different format
    s1_s = spend_real_df_1.cumsum().iloc[-1:].transpose()
    s1_s.rename(columns={s1_s.columns[0]: "Sim1" }, inplace = True)
   
    if spending_plan_2 is not '':
        s2_s = spend_real_df_2.cumsum().iloc[-1:].transpose()
        s2_s.rename(columns={s2_s.columns[0]: "Sim2" }, inplace = True)
    if spending_plan_3 is not '':
        s3_s = spend_real_df_3.cumsum().iloc[-1:].transpose()
        s3_s.rename(columns={s3_s.columns[0]: "Sim3" }, inplace = True)

    if spending_plan_2 is '':
        s_combined_s = s1_s
    elif spending_plan_3 is '':
        s_combined_s = pd.concat([s1_s,s2_s], axis=1)
    else:
        s_combined_s = pd.concat([s1_s,s2_s,s3_s], axis=1)

    # Takes the combined dataframe of ending markets values and reorganizes columns into one column
    s_combined_mv_stacked = s_combined_mv.stack().reset_index(level=0, drop=True)
    s_combined_mv_stacked = pd.DataFrame(s_combined_mv_stacked, columns = ['Ending Value'])
    s_combined_mv_stacked = s_combined_mv_stacked.reset_index()

    #  Takes the combined dataframe of $ spending values and reorganizes columns into one column
    s_combined_s_stacked = s_combined_s.stack().reset_index(drop=True)
    s_combined_s_stacked = pd.DataFrame(s_combined_s_stacked, columns = ['Total Dollars Spent'])

    # combines the two above
    s_combined_stacked_both = pd.concat([s_combined_mv_stacked, s_combined_s_stacked], axis=1)

    # Creates a scatter plot for ending MV and $ spent
    fig = px.scatter(s_combined_stacked_both, x="Ending Value", y="Total Dollars Spent", color='index', marginal_x='rug', marginal_y='rug', width=1200,height=1000, template='plotly_white',
            labels={"index": "Legend"}, opacity=0.3, trendline="ols")
    fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="right",x=0.89, font=dict(size= 20)))
    st.markdown('## Ending Portfolio Value vs. Total Dollars Spent')
    st.plotly_chart(fig)


    # Organizes dataframes for user selected percentiles
    for col in percentiles_real_1.columns[0:]:
        percentiles_real_1 = percentiles_real_1.rename(columns={col:'Sim1 Real MV '+ str(col)})

    if spending_plan_2 is not '':
        for col in percentiles_real_2.columns[0:]:
            percentiles_real_2 = percentiles_real_2.rename(columns={col:'Sim2 Real MV '+ str(col)})
    if spending_plan_3 is not '':
        for col in percentiles_real_3.columns[0:]:
            percentiles_real_3 = percentiles_real_3.rename(columns={col:'Sim3 Real MV '+ str(col)})


    # Repeat for spending percentiles
    for col in percentiles_real_spend_1.columns[0:]:
        percentiles_real_spend_1 = percentiles_real_spend_1.rename(columns={col:'Sim1 Real Spend '+ str(col)})
    if spending_plan_2 is not '':
        for col in percentiles_real_spend_2.columns[0:]:
            percentiles_real_spend_2 = percentiles_real_spend_2.rename(columns={col:'Sim2 Real Spend '+ str(col)})
    if spending_plan_3 is not '':
        for col in percentiles_real_spend_3.columns[0:]:
            percentiles_real_spend_3 = percentiles_real_spend_3.rename(columns={col:'Sim3 Real Spend '+ str(col)})

            # Organizes dataframes for user selected percentiles

    for col in percentiles_nominal_1.columns[0:]:
        percentiles_nominal_1 = percentiles_nominal_1.rename(columns={col:'Sim1 Nom MV '+ str(col)})
    if spending_plan_2 is not '':
        for col in percentiles_nominal_2.columns[0:]:
            percentiles_nominal_2 = percentiles_nominal_2.rename(columns={col:'Sim2 Nom MV '+ str(col)})
    if spending_plan_3 is not '':
        for col in percentiles_nominal_3.columns[0:]:
            percentiles_nominal_3 = percentiles_nominal_3.rename(columns={col:'Sim3 Nom MV '+ str(col)})

        # Repeat for spending percentiles
    for col in percentiles_nom_spend_1.columns[0:]:
        percentiles_nom_spend_1 = percentiles_nom_spend_1.rename(columns={col:'Sim1 Nom Spend '+ str(col)})
    if spending_plan_2 is not '':
        for col in percentiles_nom_spend_2.columns[0:]:
            percentiles_nom_spend_2 = percentiles_nom_spend_2.rename(columns={col:'Sim2 Nom Spend '+ str(col)})
    if spending_plan_3 is not '':
        for col in percentiles_nom_spend_3.columns[0:]:
            percentiles_nom_spend_3 = percentiles_nom_spend_3.rename(columns={col:'Sim3 Nom Spend '+ str(col)})


    # Plots user selected percentiles by MV

    fig = plt.figure(figsize=(10,6))
    plt.title('Market Value by Percentile')
    plt.xlabel('Qtrs')
    plt.ylabel('Portfolio Value')
    plt.plot(percentiles_real_1, color = 'royalblue', label = 'Sim 1')
    if spending_plan_2 is not '':
        plt.plot(percentiles_real_2, color = 'red', label = 'Sim 2')
    if spending_plan_3 is not '':
        plt.plot(percentiles_real_3, color = 'mediumseagreen', label = 'Sim 3')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.ticklabel_format(style='plain')
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    st.pyplot(fig)

    # Plots user selected percentiles by dollars spent

    fig = plt.figure(figsize=(10,6))
    plt.title('Quarterly Spending Power by Percentile')
    plt.xlabel('Qtrs')
    plt.ylabel('Quarterly Dollars Spent')
    plt.plot(percentiles_real_spend_1, color = 'royalblue', label = 'Sim 1')
    if spending_plan_2 is not '':
        plt.plot(percentiles_real_spend_2, color = 'red', label = 'Sim 2')
    if spending_plan_3 is not '':
        plt.plot(percentiles_real_spend_3, color = 'mediumseagreen', label = 'Sim 3')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.ticklabel_format(style='plain')
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    st.pyplot(fig)


    # Plots user selected percentiles by cumulative dollars spent

    fig = plt.figure(figsize=(10,6))
    plt.title('Cumulative Dollars Spent by Percentile')
    plt.xlabel('Qtrs')
    plt.ylabel('Cumulative Dollars Spent')
    plt.plot(percentiles_real_spend_1.cumsum(), color = 'royalblue', label = 'Sim 1')
    if spending_plan_2 is not '':
        plt.plot(percentiles_real_spend_2.cumsum(), color = 'red', label = 'Sim 2')
    if spending_plan_3 is not '':
        plt.plot(percentiles_real_spend_3.cumsum(), color = 'mediumseagreen', label = 'Sim 3')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.ticklabel_format(style='plain')
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    st.pyplot(fig)




    # Organizes percentiles into one combined dataframe to be able to download as CSV
    if spending_plan_2 is '':
        output = pd.concat([percentiles_real_1, percentiles_real_spend_1, percentiles_nominal_1, percentiles_nom_spend_1], axis=1)
    elif spending_plan_3 is '':
        output = pd.concat([percentiles_real_1, percentiles_real_spend_1, percentiles_real_2, percentiles_real_spend_2, percentiles_nominal_1, percentiles_nom_spend_1, percentiles_nominal_2, percentiles_nom_spend_2], axis=1)
    else:
        output = pd.concat([percentiles_real_1, percentiles_real_spend_1, percentiles_real_2, percentiles_real_spend_2, percentiles_real_3, percentiles_real_spend_3, percentiles_nominal_1, percentiles_nom_spend_1, percentiles_nominal_2, percentiles_nom_spend_2, percentiles_nominal_3, percentiles_nom_spend_3], axis=1)



    output_input = pd.concat([input_data, output], join = 'outer')
    # Function to download output as CSV
    @st.cache
    def convert_df(df):
    #IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv_output = convert_df(output_input)

    st.write("Click to download results as a CSV")
    st.write("CSV to include real and nominal results")
    st.download_button(
        label="Download",
        data=csv_output,
        file_name='output.csv',
        mime='text/csv',
    )



# Computes the user input results if real terms is preferred
# Returns several dataframe/list objects to be used in charting below

elif nom_check and st.button('Compute'):
    if spending_plan_1 == 'Constant (single spend rate)':
        portfolio_real_1, spend_real_df_1, portfolio_nom_1, spend_nom_df_1, percentiles_real_1, percentiles_real_spend_1, percentiles_nominal_1, percentiles_nom_spend_1 = compute_constant(annual_return_1, annual_std_1, annual_spending_1, rolling_quarters, cpi, cpi_std, t_intervals, yale_input_1, prior_qtr_spend_1)
        input_data_1 = pd.DataFrame({'Sim1': [annual_return_1,annual_std_1,annual_spending_1,'NA','NA','NA',rolling_quarters,cpi,t_intervals, yale_input_1]},
            index = ['Annual Return', 'Annual Std.', 'Annual Spend Rate (if Constant)', 'Initial Spend Rate (if Variable)','Initial Spend Rate Duration (if Variable)','Long-Term Spend Rate (if Variable)','Rolling Period (Qtrs)', 'CPI', 'Simulation Period','Yale Model (1=Yes)']
            )

    elif spending_plan_1 == 'Variable (multiple spend rates)':
        portfolio_real_1, spend_real_df_1, portfolio_nom_1, spend_nom_df_1, percentiles_real_1, percentiles_real_spend_1, percentiles_nominal_1, percentiles_nom_spend_1 = compute_variable(annual_return_1, annual_std_1, annual_spending_initial_1, annual_spending_initial_duration_1, annual_spending_final_1, rolling_quarters, cpi, cpi_std, t_intervals)
        input_data_1 = pd.DataFrame({'Sim1 Inputs': [annual_return_1,annual_std_1,'NA', annual_spending_initial_1,annual_spending_initial_duration_1,annual_spending_final_1,rolling_quarters,cpi,t_intervals,'NA']},
            index = ['Annual Return', 'Annual Std.', 'Annual Spend Rate (if Constant)', 'Initial Spend Rate (if Variable)','Initial Spend Rate Duration (if Variable)','Long-Term Spend Rate (if Variable)','Rolling Period (Qtrs)', 'CPI', 'Simulation Period','Yale Model (1=Yes)']
            )

    if spending_plan_2 == 'Constant (single spend rate)':
        portfolio_real_2, spend_real_df_2, portfolio_nom_2, spend_nom_df_2, percentiles_real_2, percentiles_real_spend_2, percentiles_nominal_2, percentiles_nom_spend_2 = compute_constant(annual_return_2, annual_std_2, annual_spending_2, rolling_quarters, cpi, cpi_std, t_intervals, yale_input_2, prior_qtr_spend_2)
        input_data_2 = pd.DataFrame({'Sim2 Inputs': [annual_return_2,annual_std_2,annual_spending_2,'NA','NA','NA',rolling_quarters,cpi,t_intervals, yale_input_2]},
            index = ['Annual Return', 'Annual Std.', 'Annual Spend Rate (if Constant)', 'Initial Spend Rate (if Variable)','Initial Spend Rate Duration (if Variable)','Long-Term Spend Rate (if Variable)','Rolling Period (Qtrs)', 'CPI', 'Simulation Period','Yale Model (1=Yes)']
            )

    elif spending_plan_2 == 'Variable (multiple spend rates)':
        portfolio_real_2, spend_real_df_2, portfolio_nom_2, spend_nom_df_2, percentiles_real_2, percentiles_real_spend_2, percentiles_nominal_2, percentiles_nom_spend_2 = compute_variable(annual_return_2, annual_std_2, annual_spending_initial_2, annual_spending_initial_duration_2, annual_spending_final_2, rolling_quarters, cpi, cpi_std, t_intervals)
        input_data_2 = pd.DataFrame({'Sim2 Inputs': [annual_return_2,annual_std_2,'NA', annual_spending_initial_2,annual_spending_initial_duration_2,annual_spending_final_2,rolling_quarters,cpi,t_intervals, 'NA']},
            index = ['Annual Return', 'Annual Std.', 'Annual Spend Rate (if Constant)', 'Initial Spend Rate (if Variable)','Initial Spend Rate Duration (if Variable)','Long-Term Spend Rate (if Variable)','Rolling Period (Qtrs)', 'CPI', 'Simulation Period','Yale Model (1=Yes)']
            )

    if spending_plan_3 == 'Constant (single spend rate)':
        portfolio_real_3, spend_real_df_3, portfolio_nom_3, spend_nom_df_3, percentiles_real_3, percentiles_real_spend_3, percentiles_nominal_3, percentiles_nom_spend_3 = compute_constant(annual_return_3, annual_std_3, annual_spending_3, rolling_quarters, cpi, cpi_std, t_intervals, yale_input_3, prior_qtr_spend_3)
        input_data_3 = pd.DataFrame({'Sim3 Inputs': [annual_return_3,annual_std_3,annual_spending_3,'NA','NA','NA',rolling_quarters,cpi,t_intervals, yale_input_3]},
            index = ['Annual Return', 'Annual Std.', 'Annual Spend Rate (if Constant)', 'Initial Spend Rate (if Variable)','Initial Spend Rate Duration (if Variable)','Long-Term Spend Rate (if Variable)','Rolling Period (Qtrs)', 'CPI', 'Simulation Period','Yale Model (1=Yes)']
            )

    elif spending_plan_3 == 'Variable (multiple spend rates)':
        portfolio_real_3, spend_real_df_3, portfolio_nom_3, spend_nom_df_3, percentiles_real_3, percentiles_real_spend_3, percentiles_nominal_3, percentiles_nom_spend_3 = compute_variable(annual_return_3, annual_std_3, annual_spending_initial_3, annual_spending_initial_duration_3, annual_spending_final_3, rolling_quarters, cpi, cpi_std, t_intervals)
        input_data_3 = pd.DataFrame({'Sim3 Inputs': [annual_return_3,annual_std_3,'NA', annual_spending_initial_3,annual_spending_initial_duration_3,annual_spending_final_3,rolling_quarters,cpi,t_intervals, 'NA']},
            index = ['Annual Return', 'Annual Std.', 'Annual Spend Rate (if Constant)', 'Initial Spend Rate (if Variable)','Initial Spend Rate Duration (if Variable)','Long-Term Spend Rate (if Variable)','Rolling Period (Qtrs)', 'CPI', 'Simulation Period','Yale Model (1=Yes)']
            )




    # Takes the real portfolio market value results and organizes the output into a dataframe for the final ending market value only
    # if statements to allow flexibility if only 1 or 2 simulations conducted

    s1_mv = pd.DataFrame(portfolio_nom_1[-1], columns=['Sim1'])
    if spending_plan_2 is not '':
        s2_mv = pd.DataFrame(portfolio_nom_2[-1], columns=['Sim2'])
    if spending_plan_3 is not '':
        s3_mv = pd.DataFrame(portfolio_nom_3[-1], columns=['Sim3'])

    # Combines each simulation into 1 dataframe
    if spending_plan_2 is '':
        s_combined_mv = s1_mv
        input_data = input_data_1
    elif spending_plan_3 is '':
        s_combined_mv = pd.concat([s1_mv,s2_mv], axis=1)
        input_data = pd.concat([input_data_1, input_data_2], axis=1)
    else:
        s_combined_mv = pd.concat([s1_mv,s2_mv,s3_mv], axis=1)
        input_data = pd.concat([input_data_1, input_data_2, input_data_3], axis=1)


    # Repeats the above process for the $ spending data
    # Code is slightly different than above given the output of the spending data being in a different format

    s1_s = spend_nom_df_1.cumsum().iloc[-1:].transpose()
    s1_s.rename(columns={s1_s.columns[0]: "Sim1" }, inplace = True)
   
    if spending_plan_2 is not '':
        s2_s = spend_nom_df_2.cumsum().iloc[-1:].transpose()
        s2_s.rename(columns={s2_s.columns[0]: "Sim2" }, inplace = True)
    if spending_plan_3 is not '':
        s3_s = spend_nom_df_3.cumsum().iloc[-1:].transpose()
        s3_s.rename(columns={s3_s.columns[0]: "Sim3" }, inplace = True)

    if spending_plan_2 is '':
        s_combined_s = s1_s
    elif spending_plan_3 is '':
        s_combined_s = pd.concat([s1_s,s2_s], axis=1)
    else:
        s_combined_s = pd.concat([s1_s,s2_s,s3_s], axis=1)

    # Takes the combined dataframe of ending markets values and reorganizes columns into one column
    s_combined_mv_stacked = s_combined_mv.stack().reset_index(level=0, drop=True)
    s_combined_mv_stacked = pd.DataFrame(s_combined_mv_stacked, columns = ['Ending Value'])
    s_combined_mv_stacked = s_combined_mv_stacked.reset_index()

    #  Takes the combined dataframe of $ spending values and reorganizes columns into one column
    s_combined_s_stacked = s_combined_s.stack().reset_index(drop=True)
    s_combined_s_stacked = pd.DataFrame(s_combined_s_stacked, columns = ['Total Dollars Spent'])

    # combines the two above
    s_combined_stacked_both = pd.concat([s_combined_mv_stacked, s_combined_s_stacked], axis=1)

    # Creates a scatter plot for ending MV and $ spent
    fig = px.scatter(s_combined_stacked_both, x="Ending Value", y="Total Dollars Spent", color='index', marginal_x='rug', marginal_y='rug', width=1200,height=1000, template='plotly_white',
            labels={"index": "Legend"}, opacity=0.3, trendline="ols")
    fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="right",x=0.89, font=dict(size= 20)))
    st.markdown('## Ending Portfolio Value vs. Total Dollars Spent')
    st.plotly_chart(fig)

    # Organizes dataframes for user selected percentiles

    for col in percentiles_nominal_1.columns[0:]:
        percentiles_nominal_1 = percentiles_nominal_1.rename(columns={col:'Sim1 Nom MV '+ str(col)})
    if spending_plan_2 is not '':
        for col in percentiles_nominal_2.columns[0:]:
            percentiles_nominal_2 = percentiles_nominal_2.rename(columns={col:'Sim2 Nom MV '+ str(col)})
    if spending_plan_3 is not '':
        for col in percentiles_nominal_3.columns[0:]:
            percentiles_nominal_3 = percentiles_nominal_3.rename(columns={col:'Sim3 Nom MV '+ str(col)})

        # Repeat for spending percentiles
    for col in percentiles_nom_spend_1.columns[0:]:
        percentiles_nom_spend_1 = percentiles_nom_spend_1.rename(columns={col:'Sim1 Nom Spend '+ str(col)})
    if spending_plan_2 is not '':
        for col in percentiles_nom_spend_2.columns[0:]:
            percentiles_nom_spend_2 = percentiles_nom_spend_2.rename(columns={col:'Sim2 Nom Spend '+ str(col)})
    if spending_plan_3 is not '':
        for col in percentiles_nom_spend_3.columns[0:]:
            percentiles_nom_spend_3 = percentiles_nom_spend_3.rename(columns={col:'Sim3 Nom Spend '+ str(col)})

            # Organizes dataframes for user selected percentiles
    for col in percentiles_real_1.columns[0:]:
        percentiles_real_1 = percentiles_real_1.rename(columns={col:'Sim1 Real MV '+ str(col)})

    if spending_plan_2 is not '':
        for col in percentiles_real_2.columns[0:]:
            percentiles_real_2 = percentiles_real_2.rename(columns={col:'Sim2 Real MV '+ str(col)})
    if spending_plan_3 is not '':
        for col in percentiles_real_3.columns[0:]:
            percentiles_real_3 = percentiles_real_3.rename(columns={col:'Sim3 Real MV '+ str(col)})

        # Repeat for spending percentiles
    for col in percentiles_real_spend_1.columns[0:]:
        percentiles_real_spend_1 = percentiles_real_spend_1.rename(columns={col:'Sim1 Real Spend '+ str(col)})
    if spending_plan_2 is not '':
        for col in percentiles_real_spend_2.columns[0:]:
            percentiles_real_spend_2 = percentiles_real_spend_2.rename(columns={col:'Sim2 Real Spend '+ str(col)})
    if spending_plan_3 is not '':
        for col in percentiles_real_spend_3.columns[0:]:
            percentiles_real_spend_3 = percentiles_real_spend_3.rename(columns={col:'Sim3 Real Spend '+ str(col)})

    

    # Plots user selected percentiles by MV
    fig = plt.figure(figsize=(10,6))
    plt.title('Market Value by Percentile')
    plt.xlabel('Qtrs')
    plt.ylabel('Portfolio Value')
    plt.plot(percentiles_nominal_1, color = 'royalblue', label = 'Sim 1')
    if spending_plan_2 is not '':
        plt.plot(percentiles_nominal_2, color = 'red', label = 'Sim 2')
    if spending_plan_3 is not '':
        plt.plot(percentiles_nominal_3, color = 'mediumseagreen', label = 'Sim 3')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.ticklabel_format(style='plain')
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    st.pyplot(fig)



    # Plots user selected percentiles by $ spent
    fig = plt.figure(figsize=(10,6))
    plt.title('Quarterly Spending Power by Percentile')
    plt.xlabel('Qtrs')
    plt.ylabel('Quarterly Dollars Spent')
    plt.plot(percentiles_nom_spend_1, color = 'royalblue', label = 'Sim 1')
    if spending_plan_2 is not '':
        plt.plot(percentiles_nom_spend_2, color = 'red', label = 'Sim 2')
    if spending_plan_3 is not '':
        plt.plot(percentiles_nom_spend_3, color = 'mediumseagreen', label = 'Sim 3')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.ticklabel_format(style='plain')
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    st.pyplot(fig)

    # Plots user selected percentiles by cumulative $ spent
    fig = plt.figure(figsize=(10,6))
    plt.title('Cumulative Dollars Spent by Percentile')
    plt.xlabel('Qtrs')
    plt.ylabel('Cumulative Dollars Spent')
    plt.plot(percentiles_nom_spend_1.cumsum(), color = 'royalblue', label = 'Sim 1')
    if spending_plan_2 is not '':
        plt.plot(percentiles_nom_spend_2.cumsum(), color = 'red', label = 'Sim 2')
    if spending_plan_3 is not '':
        plt.plot(percentiles_nom_spend_3.cumsum(), color = 'mediumseagreen', label = 'Sim 3')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.ticklabel_format(style='plain')
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    st.pyplot(fig)

    # Organizes percentiles into one combined dataframe to be able to download as CSV
    if spending_plan_2 is '':
        output = pd.concat([percentiles_real_1, percentiles_real_spend_1, percentiles_nominal_1, percentiles_nom_spend_1], axis=1)
    elif spending_plan_3 is '':
        output = pd.concat([percentiles_real_1, percentiles_real_spend_1, percentiles_real_2, percentiles_real_spend_2, percentiles_nominal_1, percentiles_nom_spend_1, percentiles_nominal_2, percentiles_nom_spend_2], axis=1)
    else:
        output = pd.concat([percentiles_real_1, percentiles_real_spend_1, percentiles_real_2, percentiles_real_spend_2, percentiles_real_3, percentiles_real_spend_3, percentiles_nominal_1, percentiles_nom_spend_1, percentiles_nominal_2, percentiles_nom_spend_2, percentiles_nominal_3, percentiles_nom_spend_3], axis=1)

    output_input = pd.concat([input_data, output], join = 'outer')
    # Function to download output as CSV
    @st.cache
    def convert_df(df):
    #IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv_output = convert_df(output_input)

    st.write("Click to download results as a CSV")
    st.write("CSV to include real and nominal results")
    st.download_button(
        label="Download",
        data=csv_output,
        file_name='output.csv',
        mime='text/csv',
    )

