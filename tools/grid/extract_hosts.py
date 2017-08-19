import sys

def check_queue():
    queues = []
    if len(sys.argv) > 1 and sys.argv[1] == '-q':
        queues = sys.argv[2].split(',')
    return queues

if __name__ == '__main__':
    allowed_queues = check_queue()
    hosts = []
    last_line = 'x x 0/0/0 x x x'
    for l in sys.stdin:
        if l.startswith('\tqf:hostname='):
            if allowed_queues and last_line.split('@')[0] not in allowed_queues:
                continue
            slots = last_line.split()[2].split('/')
            if int(slots[0]) + int(slots[1]) < int(slots[2]):
                hosts.append(l[len('\tqf:hostname='):-1])
        else:
            last_line = l
    print('\n'.join(hosts))
