#property strict

#include <Jason.mqh>
#include <Code/dataExportClass.mqh>


ENUM_TIMEFRAMES StringToTimeframe(string tf_string)
{
    if (tf_string == "PERIOD_M1") return PERIOD_M1;
    if (tf_string == "PERIOD_M5") return PERIOD_M5;
    if (tf_string == "PERIOD_M15") return PERIOD_M15;
    if (tf_string == "PERIOD_M30") return PERIOD_M30;
    if (tf_string == "PERIOD_H1") return PERIOD_H1;
    if (tf_string == "PERIOD_H4") return PERIOD_H4;
    if (tf_string == "PERIOD_D1") return PERIOD_D1;
    if (tf_string == "PERIOD_W1") return PERIOD_W1;
    return WRONG_VALUE;
}

string CheckRequiredKeys(CJAVal &data, string &required_keys[])
{
    for (int i = 0; i < ArraySize(required_keys); i++)
    {
        if (!data.HasKey(required_keys[i]))
            return required_keys[i];
    }
    return "";
}

CJAVal setErrorResponse(string message)
{
    CJAVal json;
    json["status"] = "ERROR";
    json["data"]["message"] = message;
    return json;
}

// Builds a success/error response from an export list.
// Pass trial_number >= 0 to include it in the success payload.
CJAVal BuildStatusResponse(DataExportClass &list[], int trial_number = -1)
{
    CJAVal status_json;
    bool all_success = true;
    CJAVal error_array;

    for(int i = 0; i < ArraySize(list); i++)
    {
        if (!list[i].status)
        {
            all_success = false;
            CJAVal err_obj;
            err_obj["symbol"] = list[i].symbol;
            err_obj["message"] = list[i].error_message;
            error_array.Add(err_obj);
        }
    }

    if (all_success)
    {
        status_json["status"] = "OK";
        status_json["data"]["message"] = "Data saved successfully.";
        if (trial_number >= 0)
            status_json["trial_number"] = trial_number;
    }
    else
    {
        status_json["status"] = "ERROR";
        status_json["data"]["message"] = error_array;
    }
    return status_json;
}
