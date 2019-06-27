#!/usr/bin/env python
# coding=utf-8

# creates a reverse enwiki8 dataset for a backwards transformer LM.

import os
import sys
import zipfile

PATH="/exp/dmueller/data/transformer-xl/enwik8/backward/"

if os.path.exists('{}train.txt'.format(PATH)):
    print('Tokenized enwik8 already exists - skipping processing')
    sys.exit()

data = zipfile.ZipFile('enwik8.zip').read('enwik8')

print('Length of enwik8: {}'.format(len(data)))

num_test_chars = 5000000

train_data = data[: -2 * num_test_chars]
valid_data = data[-2 * num_test_chars: -num_test_chars]
test_data = data[-num_test_chars:]

for fn, part in [('train.txt', train_data), ('valid.txt', valid_data), ('test.txt', test_data)]:
    print('{} will have {} bytes'.format(fn, len(part)))
    print('- Tokenizing...')
    part_b = map(ord, part)
    part_str = ' '.join(reversed([str(c) if c != ord('\n') else '\n' for c in part_b]))
    print('- Writing...')
    f = open(PATH + fn, 'w').write(part_str)
    f = open(PATH + fn + '.raw', 'wb').write(part[::-1])
