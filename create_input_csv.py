import json
import urllib.request
from datetime import datetime

PATH = './input.csv'                             # output file name
URL = 'https://public.bitbank.cc/btc_jpy/ticker' # Please change this url as you like

req = urllib.request.Request(URL)
writable = True


def write(time, data):
    with open(PATH, 'a') as f: # After changing above url, you also need to fix header below
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
    '''
    fetching ticker per 10 seconds
    '''

    # output file is overwritten even if it exists already
    with open(PATH, 'w') as f:
        f.write('time,sell,buy,high,low,last,vol\n') # After changing above url, you also need to fix header below

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
                data = body['data'] # After changing above url, you also need to fix this depending on your url
                write(time, data)
        elif t.second % 10 == 1 and not writable:
            writable = True

