# Crypto-Trading-Bot

cd backend
venv/Scripts/activate
py app.py

Deployement:

cd backend
source venv/bin/activate
nohup python3 app.py > flask.log 2>&1 &


view Logs:
tail -f flask.log

Check process id:

ps aux | grep app.py
kill -9 process_id