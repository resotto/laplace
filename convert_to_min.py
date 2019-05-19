import csv
import re

PATH = 'input_min.csv'
REGEX = '[\d]{4}-[\d]{2}-[\d]{2} [\d]{2}:[\d]{2}:00' # YYYY-MM-DD hh:mm:00


if __name__ == '__main__':

    # overwrite
    with open(PATH, 'w') as f:
        f.write('time,sell,buy,high,low,last,vol\n')

    pattern = re.compile(REGEX)
    val_per_min = []

    with open('input.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if pattern.match(row[0]): # row[0] equals "time"
                val_per_min.append(row)

    with open(PATH, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(val_per_min)

