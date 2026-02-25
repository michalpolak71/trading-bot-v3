#!/bin/bash
cp bot_ultimate.db /tmp/db_backup.db
python3 -c "import base64; print(base64.b64encode(open('bot_ultimate.db','rb').read()).decode())" > db_base64.txt
echo "Database exported to db_base64.txt"
