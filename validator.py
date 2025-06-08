import pandas as pd
from collections import Counter
from statsmodels.tsa.stattools import adfuller, acf
import warnings
import os
import sys

warnings.filterwarnings("ignore")

def detect_grain(dates):
    date_diffs = dates.sort_values().diff().dropna().dt.days
    if date_diffs.median() <= 1:
        return "Daily"
    elif date_diffs.median() <= 7:
        return "Weekly"
    else:
        return "Monthly"

def detect_seasonality(series):
    if len(series) < 24:
        return "No"
    autocorrs = acf(series, nlags=20, fft=False)
    peaks = [i for i in range(1, len(autocorrs)) if autocorrs[i] > 0.5]
    return "Yes" if any(peaks) else "No"

def suggest_model(series):
    if len(series) < 12:
        return "SMA"
    try:
        result = adfuller(series)
        if result[1] < 0.05:
            return "Linear"
        else:
            return "Exponential"
    except:
        return "SMA"

def run_validation(file_path):
    try:
        # Get absolute paths
        file_path = os.path.abspath(file_path)
        output_dir = os.path.dirname(file_path)
        report_path = os.path.join(output_dir, "DataValidationReport.xlsx")
        
        # Read input data
        df = pd.read_excel(file_path, sheet_name="HistoricalData")
        
        # Validation logic
        issues = []
        setting_suggestions = []
        
        # Data quality checks
        for idx, row in df.iterrows():
            row_num = idx + 2
            sku = str(row.get("SKU", "")).strip()
            date = row.get("Date")
            qty = row.get("Quantity")

            # SKU validation
            if not sku:
                issues.append([row_num, sku, date, qty, "Missing SKU"])
                
            # Date validation
            if pd.isna(date):
                issues.append([row_num, sku, date, qty, "Missing Date"])
            elif isinstance(date, pd.Timestamp):
                if date.year < 2000 or date.year > 2099:
                    issues.append([row_num, sku, date, qty, "Date out of valid range"])
            else:
                issues.append([row_num, sku, date, qty, "Invalid Date format"])

            # Quantity validation
            if pd.isna(qty):
                issues.append([row_num, sku, date, qty, "Missing Quantity"])
            elif not isinstance(qty, (int, float)):
                issues.append([row_num, sku, date, qty, "Quantity is not numeric"])
            elif qty < 0:
                issues.append([row_num, sku, date, qty, "Negative Quantity"])

        # Group-level checks
        grouped = df.groupby("SKU")
        for sku, group in grouped:
            if group["Quantity"].count() < 4:
                issues.append(["-", sku, "-", "-", "Less than 4 data points"])
            elif (group["Quantity"] == 0).all():
                issues.append(["-", sku, "-", "-", "All quantities are zero"])

            # Analysis suggestions
            grain = detect_grain(group["Date"])
            hist_window = "6M" if len(group) >= 180 else "3M"
            seasonality = detect_seasonality(group["Quantity"].fillna(0))
            model = suggest_model(group["Quantity"].fillna(0))
            setting_suggestions.append([sku, grain, hist_window, seasonality, model])

        # Prepare DataFrames
        issues_df = pd.DataFrame(issues, columns=["Row", "SKU", "Date", "Quantity", "Issue"])
        
        if setting_suggestions:
            df_sugg = pd.DataFrame(setting_suggestions, 
                                 columns=["SKU", "Recommended Grain", "Historical Window", 
                                          "Seasonality Detected", "Suggested Model"])
            overall_row = pd.DataFrame([["OVERALL_RECOMMENDATION"] + df_sugg.mode().iloc[0, 1:].tolist()],
                                     columns=df_sugg.columns)
            df_sugg = pd.concat([overall_row, df_sugg], ignore_index=True)

        # Save report
        with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
            issues_df.to_excel(writer, sheet_name="DataValidationReport", index=False)
            if setting_suggestions:
                df_sugg.to_excel(writer, sheet_name="SettingSuggestions", index=False)
                
        return True
        
    except Exception as e:
        print(f"Error during validation: {str(e)}", file=sys.stderr)
        return False