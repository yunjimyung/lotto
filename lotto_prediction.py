import csv

with open('./lotto_data.csv','r') as f:
    reader = csv.reader(f)
    for row in reader:
        lst = []
        for i in range(49):
            lst.append(0)
        for j in row:
            lst[int(j)-1] = 1
        with open('./lotto_learning.csv','a') as w:
            Writer = csv.writer(w)
            Writer.writerow(lst)
