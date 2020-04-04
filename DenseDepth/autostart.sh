cd /root/MachineLearningServer
while :
do
echo "start"
python3 app.py |tee app.py.log
done
#python3 stero.py |tee app.py.log &
