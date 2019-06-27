import argparse as ap

def get_args():
    p = ap.ArgumentParser()
    p.add_argument("--sentences-file", default="dummy.txt", type=str)
    p.add_argument("--out-file", default="test.txt", type=str)

    return p.parse_args()

def main(args):

    in_f = open(args.sentences_file, 'r')
    out_f = open(args.out_file, 'w')

    for line in in_f.readlines():

        line = line.strip()
        line_bytes = map(ord, line.encode('utf-8'))
        print(line_bytes)
        out_line = ' '.join([str(c) for c in line_bytes])

        out_f.write(out_line)
        out_f.write("\n")

    in_f.close()
    out_f.close()

if __name__ == '__main__':
    ARGS = get_args()
    main(ARGS)
