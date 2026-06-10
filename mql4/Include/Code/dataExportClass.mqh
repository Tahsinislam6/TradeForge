#property strict

#include <JAson.mqh>

// FIX: Define the constant using #define outside the class
#define MAX_START_YEAR 2017 

class DataExportClass
{
private:
    
    ENUM_TIMEFRAMES timeframe;

    MqlRates rates[];

    
    int buffer_indices[];
    int bars_to_export;
    int trial_number;
    string indicator_name;
    double indicator_params[];


    string filename;
    int file_handle;
    
    bool checkSymbol(){
        if (!SymbolSelect(symbol, true))
        {
            Print("WARNING: Could not select symbol '", symbol, "' in Market Watch. Skipping this symbol.");
            error_message = "Symbol not found";
            return false;
        }
        return true;
    }

    bool getAvailableBars(){
        bars_to_export = iBars(symbol, timeframe);

        bars_to_export = MathMin(bars_to_export, 2500);

        if (bars_to_export <= 0)
        {
            Print("Error: No historical data found for ", symbol, " on timeframe ", EnumToString(timeframe));
            error_message = "No historical data found";
            return false;
        } 
        return true;
    }

    bool getRates(){
        if(CopyRates(symbol, timeframe, 0, bars_to_export, rates) <= 0){
            Print("Error copying rates for ", symbol, " on timeframe ", EnumToString(timeframe));
            error_message = "Error copying rates";
            return false;
        }
        return true;
    }

    bool checkFileHandle(){
        if (file_handle == INVALID_HANDLE)
        {
            Print("Error opening file: ", filename, " - Error code: ", GetLastError());
            status = false;
            error_message = "Error opening file";
            return false;
        }
        return true;
    }



public:
    bool status;
    string error_message;
    string symbol;
    DataExportClass() {
        symbol = "";
        timeframe = PERIOD_CURRENT;
        indicator_name = "";
        error_message = "";
        status = false;
        bars_to_export = 0;
        filename = "";
        file_handle = INVALID_HANDLE;
        ArrayResize(rates, 0);
        ArrayResize(indicator_params, 0);
        ArrayResize(buffer_indices, 0);
        trial_number = 0;
    }

    // Constructor for indicator data export
    DataExportClass(string src_symbol, ENUM_TIMEFRAMES src_timeframe, string src_indicator_name, double &src_indicator_params[], int &src_buffer_indices[], int src_trial_number)
    {
        this.symbol = src_symbol;
        this.timeframe = src_timeframe;
        this.indicator_name = src_indicator_name;
        ArrayCopy(this.indicator_params, src_indicator_params);
        ArrayCopy(this.buffer_indices, src_buffer_indices);
        this.trial_number = src_trial_number;
        status = false;
    }

    // Constructor for OHLC data export
    DataExportClass(string src_symbol, ENUM_TIMEFRAMES src_timeframe){
        this.symbol = src_symbol;
        this.timeframe = src_timeframe;
        status = false;
    }


    // Method to export indicator data
    void ExportIndicatorData(){
        if (!checkSymbol()) return;
        if (!getAvailableBars()) return;

        // Probe the indicator before opening the output file so a missing indicator
        // fails cleanly without creating an empty CSV.
        ResetLastError();
        iCustom(symbol, timeframe, indicator_name,
            indicator_params[0], indicator_params[1], indicator_params[2], indicator_params[3],
            indicator_params[4], indicator_params[5], indicator_params[6], indicator_params[7],
            indicator_params[8], indicator_params[9], 0, 0);
        int indicator_error = GetLastError();
        if (indicator_error != 0)
        {
            error_message = StringFormat("Indicator failed to load: '%s' (error %d)", indicator_name, indicator_error);
            return;
        }

        filename = StringFormat("%s_%s_%s_%d.csv", symbol, indicator_name, IntegerToString(timeframe), trial_number);

        file_handle = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
        if (!checkFileHandle()) return;

        int bufferCount = ArraySize(buffer_indices);
        string all_data_content = "";

        string header = "DateTime";
        for (int i = 0; i < bufferCount; i++)
            header += ",Buffer_Value_" + IntegerToString(i);
        all_data_content += header + "\n";

        int digits = (int)MarketInfo(symbol, MODE_DIGITS);

        for (int i = 0; i < bars_to_export; i++)
        {
            datetime barTime = iTime(symbol, timeframe, i);
            if (barTime == 0) break; // History buffer not loaded — stop here
            if (TimeYear(barTime) < MAX_START_YEAR) continue;

            string dataLine = TimeToString(barTime, TIME_DATE | TIME_MINUTES);
            for (int j = 0; j < bufferCount; j++)
            {
                double indicatorValue = iCustom(symbol, timeframe, indicator_name,
                    indicator_params[0], indicator_params[1], indicator_params[2], indicator_params[3],
                    indicator_params[4], indicator_params[5], indicator_params[6], indicator_params[7],
                    indicator_params[8], indicator_params[9], (double)buffer_indices[j], (double)i);

                dataLine += "," + DoubleToString(indicatorValue, digits);
            }
            all_data_content += dataLine + "\n";
        }

        int bytes_written = FileWriteString(file_handle, all_data_content);
        FileClose(file_handle);
        if (bytes_written < 0)
        {
            error_message = "Error writing to file: " + filename;
            return;
        }
        Print("Successfully saved indicator data to ", filename);
        status = true;
    }

    void ExportOHLCData(){
        if (!checkSymbol()) return;
        if (!getAvailableBars()) return;
        if (!getRates()) return;
        
    filename = symbol + "_" + IntegerToString(timeframe) + ".csv";
        file_handle = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
    if (!checkFileHandle()) return;

        // Write the CSV header
        FileWrite(file_handle, "DateTime", "Open", "High", "Low", "Close", "Volume");
    int rateSize = ArraySize(rates) - 1;
        int digits = (int)MarketInfo(symbol, MODE_DIGITS);
        
        // Loop through the data (oldest to newest index)
        for(int j = 0; j < rateSize; j++)
        {
            // Use continue logic (same as original)
            if (TimeYear(rates[j].time) < MAX_START_YEAR) 
            {
                continue; // Skip bar older than 2017
            }
            
            FileWrite(file_handle, 
                TimeToString(rates[j].time, TIME_DATE | TIME_MINUTES),
                DoubleToString(rates[j].open, digits),
                DoubleToString(rates[j].high, digits),
                DoubleToString(rates[j].low, digits),
                DoubleToString(rates[j].close, digits),
                IntegerToString(rates[j].tick_volume));
        }

        FileClose(file_handle);
        Print("Successfully saved ", rateSize, " bars of data to ", filename);
        status = true;
    }

};