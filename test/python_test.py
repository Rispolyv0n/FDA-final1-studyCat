test = [[] for _ in range(10)]

print(test)
test[5].append(1)
print(test)



test2 = [0 for _ in range(10)]

print(test2)
test2[5] += 2
print(test2)



test3 = [dict() for _ in range(10)]

print(test3)
test3[2]['hi'] = 'hello'
test3[2]['num'] = 2
test3[3]['num'] = 3
print(test3)



import numpy as np
test4 = np.array([1,2,3,4,5])
test5 = np.array([5,4,3,2,1])
print(test4/test5)


import datetime
test6 = datetime.datetime.fromtimestamp(round(1556020111033/1000))
print(test6)
print(test6.year)
print(test6.month)
print(test6.day)

test7 = datetime.datetime.fromtimestamp(round(1541599458121/1000))
print(test7)
print(test7.year)
print(test7.month)
print(test7.day)

test8 = test6 - test7
print(test8)
print(test8.days)



test9 = [[3,2,1],[6,5,4],[9,8,7]]
print(test9)
for rec in test9:
    rec.sort()
print(test9)




test10 = datetime.datetime.fromtimestamp(round(1556020111033/1000))
test11 = datetime.datetime.fromtimestamp(round(1541599458121/1000))
test12 = datetime.timedelta()
test12 += (test10-test11)
print(test12)
print(test12/100)


