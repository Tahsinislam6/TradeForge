#property strict

#include <Zmq/Zmq.mqh>
#include <JAson.mqh>

// Global variables for ZMQ
Context* context = NULL;
Socket* socket = NULL;

// Function to initialize ZMQ
int OnInit()
{
    // Create the ZMQ context
    context = new Context("TradeForgeServer");
    if(context == NULL) {
        Print("Failed to create ZMQ context");
        return(INIT_FAILED);
    }
    
    // Create a socket
    socket = new Socket(context, ZMQ_REP);
    if(socket == NULL) {
        Print("Failed to create ZMQ socket");
        delete context;
        return(INIT_FAILED);
    }
    
    // Bind the socket
    if(!socket.bind("tcp://*:5555")) {
        Print("Failed to bind socket. Is the port already in use?");
        delete socket;
        delete context;
        return(INIT_FAILED);
    }

    // Set a timer to check for messages periodically
    EventSetMillisecondTimer(50);
    Print("ZMQ server initialized and listening on tcp://*:5555");
    return(INIT_SUCCEEDED);
}

// Function to deinitialize ZMQ
void OnDeinit(const int reason)
{
    EventKillTimer(); // Stop the timer

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

// Function to handle incoming messages
void OnTimer()
{
    ZmqMsg request;
    
    // Use ZMQ_DONTWAIT to check for a message without blocking
    if(socket.recv(request, ZMQ_DONTWAIT)) {
        string received_message = request.getData();
        Print("Received message from client: ", received_message);

        CJAVal reply_json_root;
        
        // Parse the message as JSON
        CJAVal json_data;
        if (!json_data.Deserialize(received_message))
        {
            reply_json_root = setStatusResponse("ERROR", "Invalid JSON format");
        }
        else
        {
            // Get the command from the JSON object
            string command = json_data["command"].ToStr();
            
            // Check for commands
            if (command == "PING")
            {   
                Print("PING command received.");
                reply_json_root = setStatusResponse("SUCCESS", "PONG");
            }
            else
            {
                reply_json_root = setStatusResponse("ERROR", "Unknown command");
            }
        }
        
        // Send the reply back to the client
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