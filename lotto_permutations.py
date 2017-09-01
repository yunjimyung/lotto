import itertools
#로또번호 경우의 수 모두 구하기....
lotto_number = []
for i in range(45):
    i+=1
    lotto_number.append(i)
#print(lotto_number)

mypermuatation =  itertools.permutations(lotto_number,6)
for j in mypermuatation:
    print (list(j))
