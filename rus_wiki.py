import os
import argparse as ap
import tarfile

DIR='/exp/scale19/data/wiki_plain_text_ru'
PATH='/exp/dmueller/scale19/lms/transformer-xl/rus/byte-data/rus_wiki/{}'

def get_args():
    p = ap.ArgumentParser()
    p.add_argument('--split', type=str, choices=['train', 'valid'])
    p.add_argument('--direction', default='fwd', type=str, choices=['fwd', 'bwd'])
    p.add_argument('--valid-percent', default=0.05, type=float)
    return p.parse_args()

def main(args):
    data = []
    out_dir = PATH.format(args.direction)

    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(DIR):
        abs_fname = "{}/{}".format(DIR, fname)
        print('opening {}'.format(abs_fname))
        f = tarfile.open(abs_fname)
        for member in f.getmembers():
            txt_file = f.extractfile(member)
            line = txt_file.readline()
            if args.direction == 'fwd':
                data += line
            elif args.direction == 'bwd':
                line = reversed(line)
                data = line + data
    
    print('There are {} bytes in rus_wiki'.format(len(data)))
    num_test_chars = int(len(data) * args.valid_percent)
    if args.split == 'train': 
        data = data[: -num_test_chars]
    else:
        data = data[-num_test_chars:]
    
    print('{} will have {} bytes'.format(args.split, len(data)))
    part_str = ' '.join([str(c) if c != ord('\n') else '\n' for c in data])
    f = open("{}/{}.txt".format(out_dir, args.split), 'w')
    f.write(part_str)
    f.close()
    print("Done writing! Stored in {}".format(out_dir))

if __name__ == '__main__':
    ARGS = get_args()
    print("Split {} with ratio {}".format(ARGS.split, ARGS.valid_percent))
    print("Direction {}".format(ARGS.direction))
    main(ARGS)
