""" For pre-processing our russian monolingual texts into bytes
Mono-texts are:
    wmt17
    unv1
    subtitles18
"""

import os
import argparse as ap
from file_read_backwards import FileReadBackwards

DATA_FILE='/exp/scale19/data/monotext_multidomain/ru/{}.train.raw.ru'
OUT_PATH='/expscratch/nandrews/david/small/{}/bwd'

def get_args():
    p = ap.ArgumentParser()
    p.add_argument("--mono-text", type=str, choices=['wmt17', 'unv1',
                                                     'subtitles18'])
    p.add_argument('--test-bytes', default=500000, type=int)
    p.add_argument('--train-limit', default=None, type=int)
    return p.parse_args()

def main(args):
    in_file = DATA_FILE.format(args.mono_text)
    out_dir = OUT_PATH.format(args.mono_text)
    os.makedirs(out_dir, exist_ok=True)

    cur_file = open("{}/valid.txt".format(out_dir), 'w')
    cur_txt_file = open("{}/valid.raw.txt".format(out_dir), 'w')
    c = 0
    writing_valid = True

    print("Starting reading of {}".format(in_file))
    with FileReadBackwards(in_file, encoding="utf-8") as frb:
        for line in frb:
            line_b = bytes(line, encoding='utf-8')[::-1]
            part_str = ' '.join([str(c) if c != ord('\n') else '\n' for c in line_b])

            cur_txt_file.write(line[::-1])
            cur_file.write(part_str)
            cur_file.write("\n") # idk, I guess i need this... fml
            c += len(line_b)
            
            if c >= args.test_bytes and writing_valid:
                writing_valid = False
                print('Done with test.')
                print('{} bytes in test'.format(c))
                cur_txt_file.close()
                cur_file.close()

                cur_file = open("{}/train.txt".format(out_dir), 'w')
                cur_txt_file = open("{}/train.raw.txt".format(out_dir), 'w')
                c = 0
            elif not writing_valid and c >= args.train_limit:
                break
   
    print('{} bytes in train'.format(c))
    cur_txt_file.close()
    cur_file.close()

if __name__ == '__main__':
    ARGS = get_args()
    print("Processing {}".format(ARGS.mono_text))
    print("Test byte count{}".format(ARGS.test_bytes))
    main(ARGS)
