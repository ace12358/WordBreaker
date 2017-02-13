import sys

for line in sys.stdin:
    words = line.rstrip().split(' ')
    for word in words:
        surface = word.split('/')[0]
        try:
            pos = word.split('/')[1]
            print('{0}\t{1}'.format(surface,pos))
            continue
        except:
            pass
        print('{0}\t'.format(word)) 
    print('EOS')
