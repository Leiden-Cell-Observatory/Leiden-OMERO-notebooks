#!/bin/bash

# Ensure the VNC password file exists and has correct permissions
mkdir -p /home/jovyan/.vnc
echo "$VNC_PW" | vncpasswd -f > /home/jovyan/.vnc/passwd
chmod 600 /home/jovyan/.vnc/passwd

# Start VNC server with XFCE
vncserver -kill $DISPLAY || true
vncserver $DISPLAY -geometry 1280x800 -depth 24 -localhost

# Wait for XFCE to start
sleep 2

# Start noVNC websockify to bridge VNC to WebSocket for browser access
# This will listen on port 6080, which jupyter-server-proxy will forward.
exec websockify --web /usr/share/novnc 6080 localhost:5901