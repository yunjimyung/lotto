import csv

with open('./lotto_data.csv','r') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        i+=1
        print(row)
