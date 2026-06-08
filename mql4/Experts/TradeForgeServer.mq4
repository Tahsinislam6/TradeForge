#property strict

#include <Zmq/Zmq.mqh>
#include <JAson.mqh>

#include <Code/indicator.mqh>

Context* context = NULL;
Socket* socket = NULL;

int OnInit()
{
    context = new Context("TradeForgeServer");
    if(context == NULL) {
        Print("Failed to create ZMQ context");
        return(INIT_FAILED);
    }

    socket = new Socket(context, ZMQ_REP);
    if(socket == NULL) {
        Print("Failed to create ZMQ socket");
        delete context;
        return(INIT_FAILED);
    }

    if(!socket.bind("tcp://*:5555")) {
        Print("Failed to bind socket. Is the port already in use?");
        delete socket;
        delete context;
        return(INIT_FAILED);
    }

    // Timer-driven polling avoids blocking the MT4 thread
    EventSetMillisecondTimer(50);
    Print("ZMQ server initialized and listening on tcp://*:5555");
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
    EventKillTimer();

    if(socket != NULL) {
        delete socket;
        socket = NULL;
    }
    if(context != NULL) {
        delete context;
        context = NULL;
    }
    Print("ZMQ server de-initialized.");
}

void OnTimer()
{
    ZmqMsg request;

    // ZMQ_DONTWAIT returns immediately if no message is waiting, keeping the timer non-blocking
    if(socket.recv(request, ZMQ_DONTWAIT)) {
        string received_message = request.getData();
        Print("Received message from client: ", received_message);

        CJAVal reply_json_root;
        CJAVal json_data;
        if (!json_data.Deserialize(received_message))
        {
            reply_json_root = setStatusResponse("ERROR", "Invalid JSON format");
        }
        else
        {
            string command = json_data["command"].ToStr();

            if (command == "PING")
            {
                Print("PING command received.");
                reply_json_root = setStatusResponse("SUCCESS", "PONG");
            }
            else if (command == "INDICATOR")
            {   
                Print("INDICATOR command received.");
                reply_json_root = SaveIndicatorData(json_data);
            }
            else
            {
                reply_json_root = setStatusResponse("ERROR", "Unknown command");
            }
        }

        // REP socket requires exactly one send per recv — always reached regardless of parse result
        string reply_str = reply_json_root.Serialize();
        ZmqMsg reply(reply_str);
        socket.send(reply);
        Print("Sent reply to client: ", reply_str);
    }
}

CJAVal setStatusResponse(string status, string message)
{
    CJAVal json;
    json["status"] = status;
    json["data"]["message"] = message;
    return json;
}
