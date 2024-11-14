source .venv/bin/activate

file=$1

# check if file ends with .py
if [[ $file != *.py ]]
then
  echo "File must be a python file"
  exit 1
fi

if [[ " $* " == *" -n "* ]]
then
  echo "Skip install requirements"
else
    pip install -r requirements.txt
fi

python3 $file