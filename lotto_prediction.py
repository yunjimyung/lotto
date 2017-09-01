import csv
import random

def make_data():
    with open('./lotto_data.csv','r') as f:
        reader = csv.reader(f)
        for row in reader:
            lst = []
            for i in range(51):
                lst.append(0)
            for j in row:
                lst[int(j)-1] = 1
                lst[49] = 1
            with open('./lotto_learning.csv','a',newline ='') as w:
                Writer = csv.writer(w)
                Writer.writerow(lst)

#matke prediction lotto number1
def make_number1():
    number_list = []
    while len(number_list) < 6:
         random_number = random.randrange(1,46)
         if random_number in number_list:
             continue
         number_list.append(random_number)
         number_list.sort()
    result = number_list
    return result

def make_number2():
    with open('./lotto_data.csv','r') as f:
        reader = csv.reader(f)
        temp1 = make_number1()
        for row in reader:
            temp2 = [int(i)for i in row]
            temp3 = list(set(temp1) - set(temp2))
            while len(temp3) < 3:
                temp1 = make_number1()
                temp3 = list(set(temp1) - set(temp2))
        return temp1

if __name__ == "__main__":
    result = make_number2()
    print(result)
