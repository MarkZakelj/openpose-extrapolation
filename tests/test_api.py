import requests
import json

def main():
    x = json.load(open('input.json'))
    res = requests.post('http://localhost:3000/predict', json=x)
    print(res.status_code)
    res = res.json()
    json.dump(res, open('output.json', 'w'), indent=2)
    print(res.json())
    


if __name__=='__main__':
    main()