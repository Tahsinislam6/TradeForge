//+------------------------------------------------------------------+
//|                                                Indicator.mqh     |
//|                                                     Header File  |
//|------------------------------------------------------------------|

#property strict

#include <JAson.mqh>
#include <Code/helper.mqh>
#include <Code/dataExportClass.mqh>


CJAVal SaveIndicatorData(CJAVal &data)
{
    string required_keys[] = {"symbols", "timeframe", "indicators", "trial_number"};
    string missing_key = CheckRequiredKeys(data, required_keys);
    if (missing_key != "")
        return setErrorResponse("Invalid JSON structure. Missing key: '" + missing_key + "'.");

    if (data["symbols"].type != jtARRAY)
        return setErrorResponse("'symbols' key is not an array");
    if (data["symbols"].Size() == 0)
        return setErrorResponse("'symbols' array is empty");

    if (data["indicators"].type != jtOBJ)
        return setErrorResponse("'indicators' key is not an object (JSON/map)");

    ENUM_TIMEFRAMES timeframe = (ENUM_TIMEFRAMES)StringToTimeframe(data["timeframe"].ToStr());
    if (timeframe == WRONG_VALUE)
        return setErrorResponse("Invalid timeframe specified.");

    CJAVal indicator_object = data["indicators"];
    int numSymbols    = data["symbols"].Size();
    int numIndicators = indicator_object.Size();
    if (numIndicators == 0)
        return setErrorResponse("'indicators' object is empty");
    int trial_number  = (int)data["trial_number"].ToDbl();

    DataExportClass IndicatorList[];
    ArrayResize(IndicatorList, numSymbols * numIndicators);
    int listIndex = 0;

    for (int k = 0; k < numIndicators; k++)
    {
        CJAVal indicator_data = indicator_object.children[k];
        string indicator_name = indicator_data.key;

        string sub_required_keys[] = {"buffer_values", "indicator_params"};
        string sub_missing_key = CheckRequiredKeys(indicator_data, sub_required_keys);
        if (sub_missing_key != "")
            return setErrorResponse(StringFormat("Invalid JSON structure for indicator '%s'. Missing key: '%s'", indicator_name, sub_missing_key));

        CJAVal params_array  = indicator_data["indicator_params"];
        CJAVal buffers_array = indicator_data["buffer_values"];

        double indicator_params[10];
        ArrayInitialize(indicator_params, 0.0);
        int n = MathMin(params_array.Size(), 10);
        CJAVal param_val;
        for(int j = 0; j < n; j++)
        {
            param_val = params_array[j];
            indicator_params[j] = param_val.ToDbl();
        }

        int buffer_values[];
        ArrayResize(buffer_values, buffers_array.Size());
        CJAVal buffer_val;
        for(int j = 0; j < buffers_array.Size(); j++)
        {
            buffer_val = buffers_array[j];
            buffer_values[j] = (int)buffer_val.ToDbl();
        }

        for(int i = 0; i < numSymbols; i++)
        {
            string symbol = data["symbols"][i].ToStr();
            IndicatorList[listIndex] = DataExportClass(symbol, timeframe, indicator_name, indicator_params, buffer_values, trial_number);
            IndicatorList[listIndex].ExportIndicatorData();
            listIndex++;
        }
    }

    return BuildStatusResponse(IndicatorList, trial_number);
}
