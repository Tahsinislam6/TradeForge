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