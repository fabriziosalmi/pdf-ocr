import json
import time
from collections import defaultdict
import threading

class ProgressManager:
    """
    A singleton class to manage progress tracking across different user sessions
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ProgressManager, cls).__new__(cls)
                cls._instance.progress = defaultdict(lambda: {"status": "idle", "percentage": 0, "message": "", "timestamp": time.time()})
                cls._instance.clients = defaultdict(set)
        return cls._instance
    
    def update_progress(self, session_id, percentage, status="processing", message=""):
        """Update progress for a specific session"""
        with self._lock:
            self.progress[session_id] = {
                "status": status,
                "percentage": percentage,
                "message": message,
                "timestamp": time.time()
            }
        return self.progress[session_id]
    
    def get_progress(self, session_id):
        """Get the current progress for a session"""
        with self._lock:
            return self.progress.get(session_id, {"status": "idle", "percentage": 0, "message": "", "timestamp": time.time()})
    
    def register_client(self, session_id, client):
        """Register a WebSocket client for a specific session"""
        with self._lock:
            self.clients[session_id].add(client)
    
    def unregister_client(self, session_id, client):
        """Unregister a WebSocket client for a specific session"""
        with self._lock:
            if session_id in self.clients and client in self.clients[session_id]:
                self.clients[session_id].remove(client)
    
    def broadcast_progress(self, session_id):
        """Broadcast progress updates to all registered clients for a session"""
        with self._lock:
            progress_data = self.get_progress(session_id)
            if session_id in self.clients:
                for client in set(self.clients[session_id]):
                    try:
                        client.send(json.dumps(progress_data))
                    except Exception:
                        # If sending fails, remove client
                        self.clients[session_id].remove(client)
    
    def cleanup_session(self, session_id):
        """Clean up resources after a session is complete"""
        with self._lock:
            if session_id in self.progress:
                del self.progress[session_id]
            if session_id in self.clients:
                self.clients[session_id].clear()
