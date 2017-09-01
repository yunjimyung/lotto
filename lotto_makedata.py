import csv
import random

def make_csv():
    random_list=[]
    while True:
        random_number = random.randrange(1,769)
        random_list.append(random_number)
        random_list = list(set(random_list))
        random_list.sort()
        if len(random_list) == 230:
            print(random_list)
            break

    with open('./lotto_learning.csv','r') as f:
        reader = csv.reader(f)
        j=0
        for row in reader:
            j += 1
            if j in random_list:
                with open('./lotto_testset.csv','a',newline ='') as w_test:
                    write_test = csv.writer(w_test)
                    write_test.writerow(row)
            else:
                with open('./lotto_trainingset.csv','a',newline ='') as w_training:
                    write_training = csv.writer(w_training)
                    write_training.writerow(row)

if __name__ == "__main__":
    result = make_csv()
