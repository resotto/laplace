import json
import urllib.request
from datetime import datetime

URL    = 'https://api.bitfinex.com/v1/pubticker/btcusd' # Please change this url as you like
PATH   = 'input.csv'                                    # Output file name
HEADER = 'time,bid,ask,last_price,volume'               # Csv header. After changing above url, you may need to fix this
FORMAT = '%Y-%m-%d %H:%M:%S'                            # Time column format

req = urllib.request.Request(URL)
writable = True


def write(time, data):
    header = HEADER.split(',')

    with open(PATH, 'a') as f:
        f.write(time)
        for i in range(1, len(header)):
            f.write(',')
            f.write(data[header[i]])
        f.write('\n')


if __name__ == '__main__':
    '''
    fetching ticker per 10 seconds
    '''

    # output file is overwritten even if it exists already
    with open(PATH, 'w') as f:
        f.write(HEADER + '\n')

    # Since consecutive values are required for input data,
    # this loop stops fetching if some error happens
    while True:
        dt = datetime.now()
        if dt.second % 10 == 0 and writable:
            writable = False
            time = dt.strftime(FORMAT)
            print(time)
            with urllib.request.urlopen(req) as res:
                body = json.load(res)
                write(time, body)                       # After changing above url, you also need to fix this depending on ticker response
        elif dt.second % 10 == 1 and not writable:
            writable = True

