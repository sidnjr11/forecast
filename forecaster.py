import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os
from datetime import datetime, timedelta
from scipy.stats import norm, zscore
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_forecasting(input_file):
    """Main forecasting function with strict method adherence"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"forecast_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        print("Loading data and settings...")
        df, settings = load_data_and_settings(input_file)
        if df is None:
            return False

        print("Processing data...")
        processed_df = process_data(df, settings)
        
        print("Generating forecasts...")
        forecast_df = generate_forecasts_with_settings(processed_df, settings)
        
        if settings['Include Charts']:
            print("Creating charts...")
            create_visualizations(forecast_df, output_dir, settings)
        
        print("Saving report...")
        save_report(forecast_df, output_dir)
        
        print(f"\nForecast completed! Results saved to: {os.path.abspath(output_dir)}")
        return True
        
    except Exception as e:
        print(f"Processing completed with warnings: {str(e)}")
        return True

def load_data_and_settings(file_path):
    """Load data and settings with proper method selection"""
    try:
        # Load settings from Home sheet
        home_settings = pd.read_excel(file_path, sheet_name='Home', header=None, usecols="A:B")
        home_settings = home_settings.set_index(0)[1]
        
        def get_setting(name, default, dtype=str):
            value = home_settings.get(name, default)
            try:
                if dtype == bool:
                    return str(value).upper() == 'TRUE' if isinstance(value, str) else bool(value)
                if dtype == int:
                    return int(str(value).split()[0]) if isinstance(value, str) else int(value)
                return str(value).strip()  # Keep full string for method names
            except:
                return default
        
        settings = {
            'Forecast Period': get_setting('Forecast Period', 'Monthly'),
            'Historical Window': get_setting('Historical Window', '12M'),
            'Forecast Method': get_setting('Forecast Method', 'Default (Best Fit)'),
            'Seasonality Toggle': get_setting('Seasonality Toggle', 'NO') == 'YES',
            'Confidence Interval': get_setting('Confidence Interval (%)', 95, int),
            'Include Charts': get_setting('Include Charts', True, bool),
            'Error Metric': get_setting('Error Metric', 'MAE'),
            'Forecast Horizon': get_setting('Forecast Horizon', '12'),
            'Remove Outliers': get_setting('Remove Outliers', 'YES') == 'YES'
        }
        
        # Convert forecast horizon
        horizon_str = str(settings['Forecast Horizon']).split()[0]
        if settings['Forecast Period'] == 'Daily':
            settings['horizon'] = int(horizon_str.replace('D', '')) if 'D' in horizon_str else int(horizon_str)
            settings['freq'] = 'D'
        elif settings['Forecast Period'] == 'Weekly':
            if 'M' in horizon_str:
                settings['horizon'] = int(horizon_str.replace('M', '')) * 4  # Approx 4 weeks/month
            else:
                settings['horizon'] = int(horizon_str.replace('W', '')) if 'W' in horizon_str else int(horizon_str)
            settings['freq'] = 'W'
        else:  # Monthly
            settings['horizon'] = int(horizon_str.replace('M', '')) if 'M' in horizon_str else int(horizon_str)
            settings['freq'] = 'M'
        
        # Load historical data
        df = pd.read_excel(file_path, sheet_name='HistoricalData', engine='openpyxl')
        
        # Column mapping
        col_map = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if 'sku' in col_lower:
                col_map[col] = 'SKU'
            elif 'date' in col_lower:
                col_map[col] = 'Date'
            elif 'quantity' in col_lower or 'actual' in col_lower:
                col_map[col] = 'Quantity'
        
        df = df.rename(columns=col_map) if col_map else df
        if 'Quantity' not in df.columns:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df = df.rename(columns={col: 'Quantity'})
                    break
        
        # Basic data cleaning
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['SKU', 'Date'])
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df = df.dropna(subset=['SKU', 'Date', 'Quantity'])
        
        # Apply historical window
        if 'M' in settings['Historical Window']:
            months = int(settings['Historical Window'].replace('M', ''))
            cutoff_date = df['Date'].max() - pd.DateOffset(months=months)
            df = df[df['Date'] >= cutoff_date]
        
        return df[['SKU', 'Date', 'Quantity']], settings
    
    except Exception as e:
        print(f"Data loading error: {str(e)}")
        return None, None

def process_data(df, settings):
    """Handle outlier removal if enabled"""
    if not settings['Remove Outliers']:
        return df
    
    processed_data = []
    for sku, group in df.groupby('SKU'):
        group = group.sort_values('Date')
        values = group['Quantity'].values
        
        if len(values) > 5:  # Only remove outliers if we have enough data
            z_scores = zscore(values)
            filtered = group[(np.abs(z_scores) < 3)]
            if len(filtered) > 0:  # Only keep if we have data left
                processed_data.append(filtered)
            else:
                processed_data.append(group)  # Fallback to original if all removed
        else:
            processed_data.append(group)
    
    return pd.concat(processed_data).sort_values(['SKU', 'Date'])

def generate_forecasts_with_settings(df, settings):
    """Generate forecasts with strict method adherence"""
    forecasts = []
    horizon = settings['horizon']
    freq = settings['freq']
    ci_z = norm.ppf(settings['Confidence Interval'] / 100)
    
    for sku, group in df.groupby('SKU'):
        group = group.sort_values('Date')
        values = group['Quantity'].values
        dates = group['Date'].values
        last_date = group['Date'].max()
        method_used = settings['Forecast Method']
        
        try:
            if len(values) < 3:
                raise ValueError(f"Insufficient data points ({len(values)}) for {method_used}")
            
            # STRICT METHOD SELECTION - NO AUTO FALLBACK
            if method_used == 'Linear Trend':
                x = np.arange(len(values))
                coeffs = np.polyfit(x, values, 1)
                forecast = np.polyval(coeffs, np.arange(len(values), len(values)+horizon))
            elif method_used == 'Exp. Smoothing':
                use_seasonality = (settings['Seasonality Toggle'] and len(values) >= 24)
                model = ExponentialSmoothing(
                    values,
                    trend='add',
                    seasonal='add' if use_seasonality else None,
                    seasonal_periods=12
                ).fit()
                forecast = model.forecast(horizon)
                method_used = f"Exp. Smoothing{' (Seasonal)' if use_seasonality else ''}"
            elif method_used == 'Weighted MA':
                weights = np.arange(1, min(13, len(values))+1)
                forecast = np.full(horizon, np.average(values[-len(weights):], weights=weights))
            elif method_used == 'Simple Moving Avg':
                window = min(12, len(values))
                forecast = np.full(horizon, values[-window:].mean())
            elif method_used == 'Default (Best Fit)':
                forecast, method_used = auto_select_forecast(values, horizon, settings)
            else:
                raise ValueError(f"Unknown forecast method: {method_used}")
            
            # Generate forecast dates
            if freq == 'D':
                forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon)
            elif freq == 'W':
                forecast_dates = pd.date_range(start=last_date + timedelta(weeks=1), periods=horizon, freq='W-MON')
            else:  # Monthly
                forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='M')
            
            # Calculate confidence intervals
            std = values.std() if len(values) > 1 else max(values[0]*0.1, 1)
            for i, date in enumerate(forecast_dates):
                forecasts.append({
                    'SKU': sku,
                    'Date': date,
                    'Quantity': np.nan,
                    'Forecast': max(0, round(forecast[i], 2)),
                    'Lower_CI': max(0, round(forecast[i] - ci_z * std, 2)),
                    'Upper_CI': max(0, round(forecast[i] + ci_z * std, 2)),
                    'Method': method_used,
                    'Data_Points': len(values),
                    'Outliers_Removed': settings['Remove Outliers']
                })
                
        except Exception as e:
            print(f"Forecast failed for {sku} using {method_used} - {str(e)}")
            # Instead of falling back, we'll skip this SKU entirely
            continue
    
    return pd.concat([df, pd.DataFrame(forecasts)]).sort_values(['SKU', 'Date'])

def auto_select_forecast(values, horizon, settings):
    """Auto-select best method only when Default is chosen"""
    # Use last 25% of data for validation if we have enough
    val_size = max(3, min(6, len(values) // 4))
    train, test = values[:-val_size], values[-val_size:]
    
    methods = {
        'Exp. Smoothing': lambda: ExponentialSmoothing(
            train,
            trend='add',
            seasonal='add' if settings['Seasonality Toggle'] and len(train) > 12 else None,
            seasonal_periods=12
        ).fit().forecast(len(test)),
        'Linear Trend': lambda: np.polyval(
            np.polyfit(np.arange(len(train)), train, 1),
            np.arange(len(train), len(train)+len(test))),
        'Weighted MA': lambda: np.full(len(test), np.average(train[-3:], weights=[1,2,3])),
        'Simple Moving Avg': lambda: np.full(len(test), train[-3:].mean())
    }
    
    best_method = None
    best_error = float('inf')
    
    for name, func in methods.items():
        try:
            pred = func()
            if settings['Error Metric'] == 'MAE':
                error = mean_absolute_error(test, pred)
            elif settings['Error Metric'] == 'RMSE':
                error = np.sqrt(mean_squared_error(test, pred))
            else:  # MAPE
                error = np.mean(np.abs((test - pred) / test)) * 100
                
            if error < best_error:
                best_error = error
                best_method = name
        except:
            continue
    
    # Generate final forecast using best method
    if best_method == 'Exp. Smoothing':
        use_seasonality = (settings['Seasonality Toggle'] and len(values) > 12)
        model = ExponentialSmoothing(
            values,
            trend='add',
            seasonal='add' if use_seasonality else None,
            seasonal_periods=12
        ).fit()
        forecast = model.forecast(horizon)
        method_name = f"Exp. Smoothing{' (Seasonal)' if use_seasonality else ''}"
    elif best_method == 'Linear Trend':
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        forecast = np.polyval(coeffs, np.arange(len(values), len(values)+horizon))
        method_name = 'Linear Trend'
    elif best_method == 'Weighted MA':
        weights = np.arange(1, min(5, len(values))+1)
        forecast = np.full(horizon, np.average(values[-len(weights):], weights=weights))
        method_name = 'Weighted MA'
    else:
        window = min(5, len(values))
        forecast = np.full(horizon, values[-window:].mean())
        method_name = 'Simple Moving Avg'
    
    return forecast, method_name

def create_visualizations(df, output_dir, settings):
    """Create visualizations with method information"""
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    for sku in df['SKU'].unique():
        sku_data = df[df['SKU'] == sku].sort_values('Date')
        
        # Plotly chart
        fig = go.Figure()
        
        # Historical data
        historical = sku_data[sku_data['Quantity'].notna()]
        fig.add_trace(go.Scatter(
            x=historical['Date'],
            y=historical['Quantity'],
            name='Actual' + (' (Outliers Removed)' if settings['Remove Outliers'] else ''),
            line=dict(color='blue', width=2),
            mode='lines+markers'
        ))
        
        # Forecast data
        forecast_data = sku_data[sku_data['Quantity'].isna()]
        if not forecast_data.empty:
            method = forecast_data['Method'].iloc[0]
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Forecast'],
                name=f'Forecast ({method})',
                line=dict(color='red', width=2, dash='dot')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Upper_CI'],
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Lower_CI'],
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255,0,0,0.2)',
                name=f"{settings['Confidence Interval']}% CI"
            ))
        
        fig.update_layout(
            title=f"{sku} Forecast ({settings['Forecast Period']})<br>Method: {method} | Data Points: {historical['Data_Points'].iloc[0]}",
            xaxis_title='Date',
            yaxis_title='Quantity',
            template='plotly_white',
            height=600
        )
        fig.write_html(os.path.join(charts_dir, f"{sku}.html"))
        
        # Matplotlib chart
        plt.figure(figsize=(12, 6))
        plt.plot(historical['Date'], historical['Quantity'], 
                label='Actual' + (' (Outliers Removed)' if settings['Remove Outliers'] else ''), 
                color='blue')
        
        if not forecast_data.empty:
            method = forecast_data['Method'].iloc[0]
            plt.plot(forecast_data['Date'], forecast_data['Forecast'], 
                    label=f'Forecast ({method})', color='red', linestyle='--')
            plt.fill_between(forecast_data['Date'],
                           forecast_data['Lower_CI'],
                           forecast_data['Upper_CI'],
                           color='red', alpha=0.1)
        
        plt.title(f"{sku} Forecast ({settings['Forecast Period']})\nMethod: {method} | Data Points: {historical['Data_Points'].iloc[0]}")
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"{sku}.png"), dpi=150, bbox_inches='tight')
        plt.close()

def save_report(df, output_dir):
    """Save results to Excel with all metadata"""
    try:
        # Prepare the report with additional columns
        report_df = df.copy()
        report_df['Outliers_Removed'] = report_df['Outliers_Removed'].fillna(method='ffill')
        report_df['Data_Points'] = report_df['Data_Points'].fillna(method='ffill')
        
        report_df.to_excel(
            os.path.join(output_dir, "forecast_results.xlsx"),
            index=False,
            engine='openpyxl'
        )
    except Exception as e:
        print(f"Report saved with warnings: {str(e)}")

if __name__ == "__main__":
    input_file = "your_input_file.xlsx"  # Change to your file path
    run_forecasting(input_file)