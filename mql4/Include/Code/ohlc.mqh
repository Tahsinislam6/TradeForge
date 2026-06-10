//+------------------------------------------------------------------+
//|                                                OHlc.mqh          |
//|                                                     Header File  |
//|------------------------------------------------------------------|

#property strict

#include <JAson.mqh>

#include <Code/helper.mqh>
#include <Code/dataExportClass.mqh>


CJAVal SaveOHLCVData(CJAVal &data)
{
    string required_keys[] = {"symbols", "timeframe"};
    string missing_key = CheckRequiredKeys(data, required_keys);
    if (missing_key != "")
        return setErrorResponse("Invalid JSON structure. Missing key: '" + missing_key + "'.");

    if (data["symbols"].type != jtARRAY)
        return setErrorResponse("'symbols' key is not an array");

    ENUM_TIMEFRAMES timeframe = (ENUM_TIMEFRAMES)StringToTimeframe(data["timeframe"].ToStr());
    if (timeframe == WRONG_VALUE)
        return setErrorResponse("Invalid timeframe specified.");

    DataExportClass OhlcList[];
    ArrayResize(OhlcList, data["symbols"].Size());

    for(int i = 0; i < data["symbols"].Size(); i++)
    {
        string symbol = data["symbols"][i].ToStr();
        OhlcList[i] = DataExportClass(symbol, timeframe);
        OhlcList[i].ExportOHLCData();
    }

    return BuildStatusResponse(OhlcList);
}
