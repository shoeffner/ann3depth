import sys


def calculate(memory, ratio, unit):
    """Calculates the memory * ratio and floors to the nearest integer.
    If the value is less than 1, it will be expressed in the next smaller unit,
    until a value is found which is bigger than 1.

    >>> calculate(32, 0.8, 'G')
    '25G'
    >>> calculate(32, 0.01, 'G')
    '320M'
    >>> calculate(32.32, 1, 'G')
    '32G'
    >>> calculate(24.385234, 0.00001, 'G')
    '243K'
    """
    def shift_unit_down(unit):
        units = 'TGMKBX'
        return units[units.index(unit) + 1]

    value = memory * ratio
    while value < 1 and unit != 'X':
        value *= 1000
        unit = shift_unit_down(unit)
    if unit == 'X':
        print(f'Value {value/1000}B too small!')
        exit(2)
    return f'{int(value // 1)}{unit}'


if __name__ == '__main__':
    memory, unit = float(sys.argv[1][:-1]), sys.argv[1][-1]
    ratio = float(sys.argv[2])
    print(calculate(memory, ratio, unit))
