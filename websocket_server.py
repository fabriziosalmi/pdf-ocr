import asyncio
import json
import threading
import websockets
from progress_manager import ProgressManager

# Global progress manager instance
progress_manager = ProgressManager()

async def websocket_handler(websocket, path):
    """Handle WebSocket connections for progress updates"""
    # Extract session ID from path (e.g., /progress/SESSION_ID)
    try:
        session_id = path.split('/')[-1]
        if not session_id:
            await websocket.close(1008, "Invalid session ID")
            return
        
        # Register client with the progress manager
        progress_manager.register_client(session_id, websocket)
        
        # Send initial progress
        progress = progress_manager.get_progress(session_id)
        await websocket.send(json.dumps(progress))
        
        # Keep connection open and handle client messages
        async for message in websocket:
            # Clients might send heartbeat messages to keep connection alive
            if message == "ping":
                await websocket.send("pong")
            
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        # Unregister client when connection is closed
        progress_manager.unregister_client(session_id, websocket)

async def start_websocket_server(host='0.0.0.0', port=8012):
    """Start the WebSocket server"""
    server = await websockets.serve(websocket_handler, host, port)
    print(f"WebSocket server running at ws://{host}:{port}")
    return server

def run_websocket_server():
    """Run the WebSocket server in a separate thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = loop.run_until_complete(start_websocket_server())
    try:
        loop.run_forever()
    except Exception as e:
        print(f"WebSocket server error: {e}")
    finally:
        loop.close()

def start_websocket_server_thread():
    """Start WebSocket server in a background thread"""
    thread = threading.Thread(target=run_websocket_server, daemon=True)
    thread.start()
    return thread

if __name__ == "__main__":
    run_websocket_server()
