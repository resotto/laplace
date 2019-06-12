import csv
import re

INPUT = 'input.csv'                                  # input file name
PATH = 'input_min.csv'                               # output file name
REGEX = '[\d]{4}-[\d]{2}-[\d]{2} [\d]{2}:[\d]{2}:00' # YYYY-MM-DD hh:mm:00


if __name__ == '__main__':
    '''
    converting time units of the input data from seconds to minutes.
    '''

    # output file is overwritten even if it exists already
    with open(PATH, 'w') as f:
        f.write('time,sell,buy,high,low,last,vol\n') # please change this header depending on ticker

    pattern = re.compile(REGEX)
    val_per_min = []

    with open(INPUT, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if pattern.match(row[0]): # row[0] equals "time"
                val_per_min.append(row)

    with open(PATH, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(val_per_min)

