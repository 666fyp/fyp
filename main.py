import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objs as go
from datetime import datetime, timedelta, date
from prophet import Prophet
from prophet.plot import plot_plotly
import os



# Define the path to your stock_data directory
stock_data_path = "./stock_data"


# Function to create a list of the last five consecutive years
def last_five_years():
    current_year = datetime.now().year
    return [str(current_year - i) for i in range(1, 6)]


# Function to load "Return on Equity %" data
def load_roe(stock_code):
    file_path = os.path.join(stock_data_path, stock_code, 'key_ratios.xls')
    df = pd.read_excel(file_path)
    years = last_five_years()
    roe_row = df[df.iloc[:,0].str.contains('Return on Equity %', na=False)]
    if not roe_row.empty:
        columns = [col for col in df.columns if col in years]
        roe_data = roe_row[columns].apply(pd.to_numeric, errors='coerce').values.flatten().tolist()  # Use pd.to_numeric
        return dict(zip(columns, roe_data))
    else:
        return {}


# Function to load NAVPS from the Excel file
def load_navps(stock_code, average_roe):
    file_path = os.path.join(stock_data_path, stock_code, 'key_ratios.xls')
    df = pd.read_excel(file_path)
    current_year = datetime.now().year - 1  # We want last year's data
    book_value_row = df[df.iloc[:,0].str.contains('Book Value', na=False)]
    
    if not book_value_row.empty:
        book_value_data = book_value_row[str(current_year)].values[0]
        book_value_data = pd.to_numeric(book_value_row[str(current_year)].values[0], errors='coerce')  # Use pd.to_numeric

        # Forecast NAVPS for the next 5 years using average ROE
        forecasted_navps = [book_value_data]
        for i in range(1, 6):
            forecasted_navps.append(forecasted_navps[-1] * (1 + average_roe))

        return forecasted_navps
    else:
        return {}



# Function to load Price to Book (P/B) ratio from the Excel file
def load_pb(stock_code):
    file_path = os.path.join(stock_data_path, stock_code, 'key_ratios.xls')
    df = pd.read_excel(file_path)
    pb_row = df[df.iloc[:,0].str.contains('Price/Book', na=False)]
    
    if not pb_row.empty:
        pb_data = pb_row['5-Yr'].values[0]
        pb_data = pd.to_numeric(pb_row['5-Yr'].values[0], errors='coerce')  # Use pd.to_numeric
        return pb_data
    else:
        return None




# Function to load eps from the Excel file
def load_eps(stock_code):
    file_path = os.path.join(stock_data_path, stock_code, 'income_statement.xls')
    df = pd.read_excel(file_path)
    years = last_five_years()
    basic_eps_row = df[df.iloc[:,0].str.contains('Basic EPS', na=False)]

    if not basic_eps_row.empty:
        columns = [col for col in df.columns if col in years]
        basic_eps_data = basic_eps_row[columns].values.flatten().tolist()
        basic_eps_data = basic_eps_row[columns].apply(pd.to_numeric, errors='coerce').values.flatten().tolist()  # Use pd.to_numeric
        return dict(zip(columns, basic_eps_data))
    else:
        return {}



# Function to load Price to Earnings (P/E) ratio from the Excel file
def load_pe(stock_code):
    file_path = os.path.join(stock_data_path, stock_code, 'key_ratios.xls')
    df = pd.read_excel(file_path)
    pe_row = df[df.iloc[:,0].str.contains('Price/Earnings', na=False)]
    
    if not pe_row.empty:
        pe_data = pe_row['5-Yr'].values[0]
        pe_data = pd.to_numeric(pe_row['5-Yr'].values[0], errors='coerce')  # Use pd.to_numeric
        return pe_data
    else:
        return None



# Function to fetch stock data from yfinance
def fetch_stock_data(stock_code):
    stock = yf.Ticker(stock_code)
    stock_info = stock.info
    pe_ratio = stock_info.get('trailingPE', None)
    price = stock_info.get('regularMarketPreviousClose', None)
    eps = stock_info.get('trailingEps', None)
    beta = stock_info.get('beta', None)


    return {
        'pe_ratio': pe_ratio,
        'price': price,
        'eps': eps,
        'beta': beta
    }










############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################


# Technical Analysis Part Starts Here

# Streamlit UI code
st.sidebar.title('StockSmart')

# Generate the list of stock folders
stock_list = [folder for folder in os.listdir(stock_data_path) if os.path.isdir(os.path.join(stock_data_path, folder))]


# Sidebar for stock selection
selected_stock = st.sidebar.selectbox("Enter Stock Code", stock_list).upper()

# Check if the user has entered a stock code
if selected_stock == "":
    # Display a welcome message on the main page if no stock code is entered
    st.write('Please enter a stock code in the sidebar to begin.')
else:
    # Proceed with the normal program code if a stock code is entered
    st.title(selected_stock + ' - Technical Analysis')

    # Set today's date and the max value for the start date (30 days before today)
    today = datetime.now()
    max_start_date = today - timedelta(days=30)

    # User can select a start date and an end date. The start date is limited to 30 days before today.
    start_date = st.sidebar.date_input("Select start date", datetime(2023, 6, 1), max_value=max_start_date)
    end_date = st.sidebar.date_input("Select end date", today, max_value=today)
    

    # Attempt to fetch data from yfinance
    try:
        data = yf.download(selected_stock, start=start_date, end=end_date)
        
        # Check if data is empty
        if data.empty:
            st.error('Unable to find the stock code: ' + selected_stock + '. Please enter a valid stock code.')
            

        else:
            # Calculate SMAs 10 and 20, Bollinger Bands, MACD, RSI, DMI
            data['SMA_10'] = ta.sma(data['Close'], length=10)
            data['SMA_20'] = ta.sma(data['Close'], length=20)
            data['SMA_50'] = ta.sma(data['Close'], length=50)
            data['SMA_120'] = ta.sma(data['Close'], length=120)
            macd = data.ta.macd(close='close', fast=12, slow=26, signal=9)
            data['RSI'] = ta.rsi(data['Close'])
            data[['DMI+','DMI-','ADX']] = ta.adx(data['High'], data['Low'], data['Close'])



            # Calculate Bollinger Bands
            bbands = ta.bbands(data['Close'], length=20, std=2.0)

            # Assign the Bollinger Bands to the data DataFrame correctly
            data['upper_band'] = bbands['BBU_20_2.0']  # Upper Bollinger Band
            data['middle_band'] = bbands['BBM_20_2.0']  # Middle Bollinger Band
            data['lower_band'] = bbands['BBL_20_2.0']  # Lower Bollinger Band


            # Create the candlestick chart 
            fig = go.Figure(data=[go.Candlestick(x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            increasing_line_color='green',
                            decreasing_line_color='red',
                            name='K-Bar')])  # Set the legend name

            # Add SMAs and Bollinger Bands if toggled on by the user
            if st.sidebar.checkbox('Show SMA 10', value=True):
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_10'], mode='lines', name='SMA 10', line=dict(color='skyblue')))
            if st.sidebar.checkbox('Show SMA 20', value=True):
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20', line=dict(color='pink')))
            if st.sidebar.checkbox('Show SMA 50', value=False):
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50'))
            if st.sidebar.checkbox('Show SMA 120', value=False):
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_120'], mode='lines', name='SMA 120'))
            if st.sidebar.checkbox('Show Bollinger Bands', value=False):
                fig.add_trace(go.Scatter(x=data.index, y=data['upper_band'], mode='lines', name='Upper BB', line=dict(color='skyblue')))
                fig.add_trace(go.Scatter(x=data.index, y=data['middle_band'], mode='lines', name='Middle BB', line=dict(color='pink', dash='dash')))
                fig.add_trace(go.Scatter(x=data.index, y=data['lower_band'], mode='lines', name='Lower BB', line=dict(color='skyblue')))

            # Plot the main candlestick chart
            st.plotly_chart(fig)


            # MACD plot
            if st.sidebar.checkbox('Show MACD', value=True):
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=data.index, y=macd['MACD_12_26_9'], mode='lines', name='MACD', line=dict(color='blue')))
                fig_macd.add_trace(go.Scatter(x=data.index, y=macd['MACDs_12_26_9'], mode='lines', name='Signal', line=dict(color='orange')))
                fig_macd.update_layout(title = 'Moving Average Convergence Divergence (MACD)')

                # Define colors for the histogram
                hist_colors = ['red' if val < 0 else 'green' for val in macd['MACDh_12_26_9']]

                # Add the MACD Histogram
                fig_macd.add_trace(go.Bar(x=data.index, y=macd['MACDh_12_26_9'], name='Histogram', marker=dict(color=hist_colors)))

                st.plotly_chart(fig_macd)


            # RSI plot
            if st.sidebar.checkbox('Show RSI', value=True):
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'))

                # Add horizontal lines for RSI levels 70, 50, and 30
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", annotation_position="bottom right")
                fig_rsi.add_hline(y=50, line_dash="dash", line_color="grey", annotation_text="Neutral (50)", annotation_position="bottom right")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)", annotation_position="bottom right")

                # Update layout to show the horizontal lines better
                fig_rsi.update_layout(
                    title='Relative Strength Index (RSI)',
                    yaxis_title='RSI',
                    xaxis_title='Date',
                    showlegend=True
                )

                # Plot RSI chart
                st.plotly_chart(fig_rsi)


            # DMI plot
            if st.sidebar.checkbox('Show DMI', value=True):
                fig_dmi = go.Figure()
                fig_dmi.add_trace(go.Scatter(x=data.index, y=data['DMI+'], mode='lines', name='DMI+', line=dict(color='blue')))
                fig_dmi.add_trace(go.Scatter(x=data.index, y=data['DMI-'], mode='lines', name='DMI-', line=dict(color='Orange')))
                fig_dmi.add_trace(go.Scatter(x=data.index, y=data['ADX'], mode='lines', name='ADX', line=dict(color='red')))
                fig_dmi.update_layout(title = 'Directional Movement Index (DMI)')
                                
                # Plot DMI chart
                st.plotly_chart(fig_dmi)

            st.sidebar.write("\n")
            st.sidebar.write("\n")





            ############################################################################################################
            ############################################################################################################
            ############################################################################################################
            ############################################################################################################


            # Valuation Model Part Starts here


            # Fair Valuation Model selection
            st.write('\n')
            st.write('\n')
            st.title(selected_stock + ' - Fair Price Valuation')


            # Define your options for the selectbox
            options = ["ROE Valuation Model", "EPS Valuation Model", "PEG Valuation Calculator"]

            # Add a default option at the beginning of the list
            options = ["Please Select"] + options

            # Create the selectbox with the default option
            fair_valuation_model = st.selectbox("Model Options:", options)

            st.write('\n')


            # Running different parts of the code based on the user's selection of the Fair Valuation Model
            if selected_stock:

                # IF select PEG Model
                if fair_valuation_model == 'PEG Valuation Calculator':

                    # Fetch data using yfinance
                    stock_data = fetch_stock_data(selected_stock)

                    # Create columns for selected stock and S&P 500
                    col1, col2, = st.columns(2)

                    # Column for S&P 500
                    with col1:
                        st.subheader('S&P 500 Index')

                        # User inputs for PE and G for S&P 500
                        sp500_pe = st.number_input('PE' + '.   '+ '[View SPX PE](https://www.gurufocus.com/economic_indicators/57/pe-ratio-ttm-for-the-sp-500)', value=28.36, format="%.2f")
                        st.write('\n\n')
                        sp500_g = st.number_input('G (% YoY)' + '.   '+ '[View SPX EPS Growth](https://www.gurufocus.com/economic_indicators/4281/sp-500-eps-with-estimate-ttm)', value=12.79, format="%.2f")
                        st.write('\n\n') 
                        st.write('\n\n')
                        sp500_peg = st.number_input('Index PEG', value= sp500_pe/sp500_g, format="%.2f")


                    # Column for the selected stock
                    with col2:
                        st.subheader(selected_stock)

                        # link for user to get estimate data
                        markdown_link = f"[{selected_stock} Stock Summary](https://www.gurufocus.com/stock/{selected_stock}/summary)"
                        st.write('View Analyst Estimate:  ' + markdown_link)

                        # Display the fetched data
                        st.write(f"Price: {stock_data['price']}")
                        st.write(f"Beta: {stock_data['beta']}")

                        pe_ratio = stock_data['pe_ratio']
                        st.write(f"PE: {pe_ratio:.2f}")

                        st.write(f"EPS: {stock_data['eps']}")
                        st.write('\n\n')

                        estimate_epsg = st.number_input('Future EPS Growth %', value= 1.0, format="%.2f")


                        # Calculation Model
                        st.write('\n\n')
                        stock_peg = stock_data['pe_ratio'] / estimate_epsg
                        st.write("Stock PEG: " + "{:.2f}".format(stock_peg))

                        fair_pe = stock_data['beta'] * sp500_pe
                        st.write("Fair PE: " + "{:.2f}".format(fair_pe))

                        fair_peg = stock_data['beta'] * sp500_peg
                        st.write("Fair PEG: " + "{:.2f}".format(fair_peg))
                        
                        pe_fair_price = stock_data['price'] * (fair_pe / stock_data['pe_ratio'])
                        peg_fair_price = stock_data['price'] * (fair_peg / stock_peg)
                        

                        # Display result in table
                        fair_prices_df = pd.DataFrame({
                        'Metric': ['PE Fair Price', 'PEG Fair Price'],
                        'Value': [f"${pe_fair_price:.2f}", f"${peg_fair_price:.2f}"]})

                        fair_prices_df.set_index('Metric', inplace=True)
                        st.table(fair_prices_df)        
                
                
                # stock in stocklist then can also do ROE / EPS Model
                elif selected_stock in stock_list:

                    # IF Select ROE Model
                    if fair_valuation_model == 'ROE Valuation Model':

                        # Displaying Return on Equity data table
                        roe = load_roe(selected_stock)

                        if roe:
                            # Convert the ROE dictionary to a DataFrame with years as columns and 'ROE' as row header
                            roe_df = pd.DataFrame(list(roe.items()), columns=['Year', 'Return on Equity %']).set_index('Year').T


                            # Display the ROE DataFrame in Streamlit with years as column headers
                            st.write("\n\n")
                            st.subheader('Return on Equity % Historical Data')
                            st.table(roe_df.style.format("{:.2f}"))
                            
                            # Calculate the average of the Return on Equity % data
                            average_roe = roe_df.loc['Return on Equity %', :].mean()

                            # Display the average below the ROE table
                            st.write(f"Average ROE Growth Rate: {average_roe:.2f}%")
                            st.write("\n\n")
                        else:
                            st.write("Return on Equity % data not found for the selected stock.")





                        # Displaying NAVPS data table
                        roe = load_roe(selected_stock)

                        if roe:
                            average_roe = pd.DataFrame(list(roe.items()), columns=['Year', 'Return on Equity %']).set_index('Year').T.loc['Return on Equity %', :].mean() / 100
                            navps_data = load_navps(selected_stock, average_roe)
                            
                            if navps_data:
                                current_year = datetime.now().year - 1
                                navps_years = [f"T+{i}" for i in range(6)]  # ["T", "T+1", "T+2", "T+3", "T+4", "T+5"]
                                navps_years[0] = "Current Year T"
                                
                                # Create a DataFrame for NAVPS values
                                navps_df = pd.DataFrame([navps_data], columns=navps_years, index=['NAVPS / Book Value'])
                                
                                # Display the NAVPS DataFrame in Streamlit
                                st.subheader('NAVPS / Book Value Forecast')
                                st.table(navps_df.style.format("{:.2f}"))
                                st.write("\n\n")

                            else:
                                st.write("NAVPS / Book Value data not found for the selected stock.")
                        else:
                            st.write("Return on Equity % data not found for the selected stock.")



                        # Displaying P/B data and calculate Fair Value table
                        pb_data = load_pb(selected_stock)

                        if pb_data is not None:
                            # Calculate and display Fair Value table
                            navps_data = load_navps(selected_stock, average_roe)
                            if navps_data:
                                # Calculate Fair Value for T+1 to T+5
                                fair_value_data = [pb_data * navps for navps in navps_data[1:]]  # Skip current year NAVPS (index 0)

                                # Display the 5-Year Average P/B Ratio and the Fair Value Forecast
                                st.subheader('Fair Value Forecast')
                                st.write(f"5-Year Average P/B Ratio: {pb_data}")

                                # Create a DataFrame for Fair Value
                                fair_value_years = [f"T+{i}" for i in range(1, 6)]
                                fair_value_df = pd.DataFrame([fair_value_data], columns=fair_value_years, index=['Fair Value'])
                                
                                # Display the Fair Value DataFrame in Streamlit
                                st.table(fair_value_df.style.format("${:.2f}"))
                            else:
                                st.write("NAVPS / Book Value data not found for the selected stock.")
                        else:
                            st.write("Price/Book (P/B) ratio data not found for the selected stock.")



                    # IF select EPS Model
                    elif fair_valuation_model == 'EPS Valuation Model':

                        # Display Basic EPS data table
                        basic_eps = load_eps(selected_stock)
                        
                        if basic_eps:
                            # Convert the Basic EPS dictionary to a DataFrame with years as columns and 'Basic EPS' as row header
                            basic_eps_df = pd.DataFrame(list(basic_eps.items()), columns=['Year', 'Basic EPS']).set_index('Year').T
                            
                            # Display the Basic EPS DataFrame in Streamlit with years as column headers
                            st.write("\n\n")
                            st.subheader('Basic EPS Historical Data')
                            st.table(basic_eps_df.style.format("{:.2f}"))

                            # Perform the calculation ((A/B)^0.25 -1) where A is current year - 1 EPS and B is current year - 5 EPS
                            try:
                                A = float(basic_eps[str(datetime.now().year - 1)])
                                B = float(basic_eps[str(datetime.now().year - 5)])
                                eps_growth_rate = ((A / B) ** 0.25) - 1

                                # Display the result of EPS growth rate
                                st.write(f"EPS Growth Rate: {eps_growth_rate:.2%}")

                                # Prepare data for the EPS Forecast table
                                last_known_eps = basic_eps[str(datetime.now().year - 1)]  # Get the last known EPS
                                forecast_values = [last_known_eps * ((1 + eps_growth_rate) ** i) for i in range(6)]

                                # Create the EPS Forecast DataFrame with years as columns and 'EPS' as a row header
                                eps_forecast_df = pd.DataFrame({
                                    'Year': ['Current Year T'] + [f"T+{i}" for i in range(1, 6)],
                                    'EPS': forecast_values})
                            

                                # Reorient the DataFrame to have years as column headers and 'EPS' as the row header
                                eps_forecast_df = eps_forecast_df.set_index('Year').T

                                # Display the EPS Forecast DataFrame in Streamlit
                                st.write("\n\n")
                                st.subheader('EPS Forecast')
                                st.table(eps_forecast_df.style.format("{:.2f}"))


                            except KeyError as e:
                                st.write("Error: Could not find EPS data for the required years for growth calculation.")
                            except Exception as e:
                                st.write(f"An error occurred during the growth calculation: {e}")


                            pe_ratio = load_pe(selected_stock)  
                            if pe_ratio is not None:
                                    # Calculate Fair Value for T+1 to T+5 using the P/E ratio and the forecasted EPS
                                    fair_value_data = [pe_ratio * eps for eps in forecast_values[1:]]  # Skip current year EPS (index 0)

                                    # Display the P/E Ratio and the Fair Value Forecast
                                    st.write("\n\n")
                                    st.subheader('Fair Value Forecast')
                                    st.write(f"5-Year Average P/E Ratio: {pe_ratio}")

                                    # Create a DataFrame for Fair Value
                                    fair_value_years = [f"T+{i}" for i in range(1, 6)]
                                    fair_value_df = pd.DataFrame([fair_value_data], columns=fair_value_years, index=['Fair Value'])

                                    # Display the Fair Value DataFrame in Streamlit
                                    st.table(fair_value_df.style.format("${:.2f}"))
                            else:
                                st.write("P/E ratio data not found for the selected stock.")
                        else:
                            st.write("Basic EPS data not found for the selected stock.")    



                else: 
                    st.write("No Fundamental Data for " + selected_stock + ". Only PEG Calculator is available!")



            ###########################################################################################################
            ############################################################################################################
            ############################################################################################################
            ############################################################################################################
            
            # Forecasting feature starts here

            # Check if the 'toggle' key exists in session_state, if not initialize it as False
            if 'forecast' not in st.session_state:
                st.session_state['forecast'] = False

            # When the button is clicked, toggle the 'toggle' state
            if st.sidebar.button('Run / Stop Price Forecasting'):
                st.session_state['forecast'] = not st.session_state['forecast']


            # Based on the current state of 'toggle', show or hide the feature
            if st.session_state['forecast']:

                st.write('\n')
                st.write('\n')
                st.write('\n')
                st.write('\n')
                st.title(selected_stock + " - Price Forecasting")
                st.sidebar.write('Forecast Status: Running...')

                # Date Range
                START = "2017-01-01"
                TODAY = date.today().strftime("%Y-%m-%d")


                n_years = st.slider('Years of prediction:', 1, 3)
                period = n_years * 365

                @st.cache_data
                def load_data(ticker):
                    data = yf.download(ticker, START, TODAY)
                    data.reset_index(inplace=True)
                    return data


                data_load_state = st.text('Loading data...')
                data = load_data(selected_stock)
                data_load_state.text('Loading data... done!')

                # Predict forecast with Prophet.
                df_train = data[['Date','Close']]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

                m = Prophet()
                m.fit(df_train)
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)

                # Show and plot forecast
                st.write('\n')
                st.subheader(f'Forecast plot for {n_years} years')
                fig1 = plot_plotly(m, forecast)
                st.plotly_chart(fig1)

                st.subheader('Forecast data')
                st.write(forecast.tail())

                st.write('\n')
                st.subheader("Forecast components")
                fig2 = m.plot_components(forecast)
                st.write(fig2)


            else:
                st.sidebar.write('Forecast Status: Stop')


    except Exception as e:
        st.error('Error fetching stock data for {}: {}'.format(selected_stock, str(e)))
        # End the script early since there's no data to display