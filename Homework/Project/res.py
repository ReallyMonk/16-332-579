with open('F:\Rutgers\\2ndSemester\Intro to DL\Homework\Project\\train.txt') as f:
    for line in f.readlines():
        acc_0 = line.split(':')[1]
        acc = acc_0.split('\\')[0]
        print(acc)
