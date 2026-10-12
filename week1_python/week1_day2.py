import math
'''
r = 2.5
s = 3.14 * r ** 2
print(f'The area of the circle with radius {r:.2f} is {s:.2f}')

s1 = 89
s2 = 85
s3 = (s2 - s1) / s1 * 100
print(f'The score difference is {s3:.2f}%')


#list
list = [1,2,3,'ye','chu']
a = len(list)
b =list[-1] #统计个数，5个
print(f'a={a},b={b}')
list.append('new')#在列表末尾添加元素
print(list)
#insert
list.insert(2,'love')#在索引为2插入love
print(list)
#pop()删除最后一个元素，pop(i)删除指定位置
list.pop()#删除最后一个元素
print(list)
list1 = [1,4,7,list]#列表嵌套
print(list1[3][4])#可以看作是二维数组，输出ye


#tuple
list2 = [1,2,3]
t1 = (1,)#1个元素必须这样
print(t1)
t2 =(8,9,list2)#元组嵌套
print(t2)
list2.append(4)
print(t2)#元组内的列表可以改变，输出(8,9,[1,2,3,4])

L = [
    ['Apple', 'Google', 'Microsoft'],
    ['Java', 'Python', 'Ruby', 'PHP'],
    ['Adam', 'Bart', 'Bob']
]

# 打印Apple:
print(L[0][0])
# 打印Python:
print(L[1][1])
# 打印Bob:
print(L[2][2])


#dict
d = {'ye':89,'chu':88,'peng':99}#ye是key，89是value
print(d['peng'])#输出99
print('chu' in d)#判断chu是否在字典中，输出True
print(d.get('chu',-1)) #如果key不存在，返回-1
#d.pop('chu')#删除‘chu’对应的键值对

#set
s = set([1,2,3,4,5,6])
print(s)
s.add(7)#添加元素
s.remove(3)
print(s)
s2 = {1,4,7,8}
print(s & s2)#交集
print(s | s2)#并集

def abs(x):
    if x>0:
        return x
    else:
        pass

print(abs(-5))

def quadratic(a, b, c):
    delta = b**2-4*a*c
    if(delta<0):
        return none
    else:
        x1=(-b+math.sqrt(delta))/(2*a)
        x2=(-b-math.sqrt(delta))/(2*a)
        return (x1, x2)

# 测试:
print('quadratic(2, 3, 1) =', quadratic(2, 3, 1))
print('quadratic(1, 3, -4) =', quadratic(1, 3, -4))

if quadratic(2, 3, 1) != (-0.5, -1.0):
    print('测试失败')
elif quadratic(1, 3, -4) != (1.0, -4.0):
    print('测试失败')
else:
    print('测试成功')

def add_end(L=[]):
    L.append('END')
    return L
add_end()
add_end()
print(add_end())#输出['END','END','END']，因为默认参数L指向同一个list   

def calc(*numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum
print(calc(1,2,3))
list = [1,2,3]
print(calc(*list))#把list变为可变参数

def mul(x, *y):
    a=1
    for i in y:
        a = a * i
    return x * a

# 测试
print('mul(5) =', mul(5))
print('mul(5, 6) =', mul(5, 6))
print('mul(5, 6, 7) =', mul(5, 6, 7))
print('mul(5, 6, 7, 9) =', mul(5, 6, 7, 9))
if mul(5) != 5:
    print('mul(5)测试失败!')
elif mul(5, 6) != 30:
    print('mul(5, 6)测试失败!')
elif mul(5, 6, 7) != 210:
    print('mul(5, 6, 7)测试失败!')
elif mul(5, 6, 7, 9) != 1890:
    print('mul(5, 6, 7, 9)测试失败!')
else:
    try:
        mul()
        print('mul()测试失败!')
    except TypeError:
        print('测试成功!')

from functools import reduce

def str2num(s):
    return float(s)

def calc(exp):
    ss = exp.split('+')
    ns = map(str2num, ss)
    return reduce(lambda acc, x: acc + x, ns)

def main():
    r = calc('100 + 200 + 345')
    print('100 + 200 + 345 =', r)
    r = calc('99 + 88 + 7.6')
    print('99 + 88 + 7.6 =', r)

main()
'''
fpath = 'D:\\Desktop\\1.txt'
with open(fpath,'r') as f:
    s=f.read()
    print(s)
