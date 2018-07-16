import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    parser.add_argument('--iterations', type=float, default=100)
    options = parser.parse_args()

    import pdb; pdb.set_trace()
    with open(options.file, 'r') as fin:
        lines = fin.readlines()
        avg_acc = 0
        for idx in range(options.iterations):
            avg_acc += float(lines[-(idx+1)]) / options.iterations
        print(avg_acc)
