
#!/bin/bash
mkdir -p venv
virtualenv --system-site-packages -p python3 ./venv
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ./venv/bin/activate && pip install -r $DIR/requirements.txt