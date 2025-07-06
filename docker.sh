!/bin/bash

mkdir -p /var/py_workspace && cd /var/py_workspace
git clone git@github.com:zhao458114067/xu_ai_assistant.git && cd xu_ai_assistant

docker build -t xu_ai_assistant .

docker run -it -v /vector_repo:/vector_repo -v /vector_store:/app/vector_store -v /var/py_workspace/xu_ai_assistant:/app --network host xu_ai_assistant /bin/bash