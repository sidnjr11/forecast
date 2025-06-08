import sys
from validator import run_validation
from forecaster import run_forecasting

def main():
    if len(sys.argv) < 3:
        print("Usage: forecast_tool.exe <mode> <input_file.xlsx>")
        print("Modes: validate, forecast")
        sys.exit(1)

    mode = sys.argv[1].lower()
    file_path = sys.argv[2]

    if mode == "validate":
        success = run_validation(file_path)
        sys.exit(0 if success else 1)

    elif mode == "forecast":
        run_forecasting(file_path)
        sys.exit(0)  # Always succeed unless exceptions are fatal

    else:
        print(f"Unknown mode '{mode}'. Use 'validate' or 'forecast'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
