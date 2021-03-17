array = []



a = '1'
for i in [1, 4, 7]:
    i = i-1
    array.append(['0'] * 9)

    array[-1][i] = a
    array[-1][i+1] = a
    array[-1][i+2] = a

    
for i in range(1, 4):
    i = i-1
    array.append(['0'] * 9)

    array[-1][i] = a
    array[-1][i+3] = a
    array[-1][i+6] = a

array.append(['0'] * 9)

array[-1][0] = a
array[-1][4] = a
array[-1][8] = a

array.append(['0'] * 9)

array[-1][2] = a
array[-1][4] = a
array[-1][6] = a

print("[")
for i in array:
    print('[' + ', '.join(reversed(i)) + '],')
print(']')