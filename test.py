
import requests
r = requests.get("http://192.168.0.185/")
print(r.status_code)