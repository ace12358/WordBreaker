import sys

golds = list()
kyteas = list()
nns = list()

for line in open(sys.argv[1]):
    line = line.rstrip()
    golds.append(line)

for line in open(sys.argv[2]):
    line = line.rstrip()
    kyteas.append(line)

for line in open(sys.argv[3]):
    line = line.rstrip()
    nns.append(line)

for i in range(len(golds)):
    #print('line:', i)
    gold = golds[i]
    kytea = kyteas[i]
    nn = nns[i]
    if not gold == kytea and not gold == nn:
        print('both results are wrong:\ngold:\t{0}\nkytea:\t{1}\nnn:\t{2}'.format(gold, kytea, nn))
    elif  gold == kytea and not gold == nn:
        print('nn result is only wrong:\ngold:\t{0}\nkytea:\t{1}\nnn:\t{2}'.format(gold, kytea, nn))
    elif not gold == kytea and gold == nn:
        print('kytea result is only wrong:\ngold:\t{0}\nkytea:\t{1}\nnn:\t{2}'.format(gold, kytea, nn))
    #else:
    #    print('both results are true!')
    
