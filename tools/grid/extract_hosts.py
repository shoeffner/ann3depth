import sys

if __name__ == '__main__':
    hosts = []
    last_line = 'x x 0/0/0 x x x'
    for l in sys.stdin:
        if l.startswith('\tqf:hostname='):
            slots = last_line.split()[2].split('/')
            if int(slots[0]) + int(slots[1]) < int(slots[2]):
                hosts.append(l[len('\tqf:hostname='):-1])
        else:
            last_line = l
    print('\n'.join(hosts))
