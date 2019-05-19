import json
import urllib.request
from datetime import datetime

PATH = './input.csv'
URL = 'https://public.bitbank.cc/btc_jpy/ticker'

req = urllib.request.Request(URL)
writable = True


def write(time, data):
    with open(PATH, 'a') as f:
        f.write(time)
        f.write(',')
        f.write(data['sell'])
        f.write(',')
        f.write(data['buy'])
        f.write(',')
        f.write(data['high'])
        f.write(',')
        f.write(data['low'])
        f.write(',')
        f.write(data['last'])
        f.write(',')
        f.write(data['vol'])
        f.write('\n')


if __name__ == '__main__':

    # overwrite
    with open(PATH, 'w') as f:
        f.write('time,sell,buy,high,low,last,vol\n')

    # Since consecutive values are required for input data,
    # this loop stops fetching if some error happens
    while True:
        t = datetime.now()
        # get ticker per 10 seconds
        if t.second % 10 == 0 and writable:
            writable = False
            time = t.strftime('%Y-%m-%d %H:%M:%S')
            print(time)
            with urllib.request.urlopen(req) as res:
                body = json.load(res)
                data = body['data']
                write(time, data)
        elif t.second % 10 == 1 and not writable:
            writable = True

