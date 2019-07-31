from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from functools import partial

from collections import Counter, OrderedDict
import pickle
import json
import multiprocessing as mp

import numpy as np

from absl import flags
import tensorflow as tf
from vocabulary import Vocab

from tensorflow.gfile import Exists as exists
from tensorflow.gfile import MakeDirs as makedirs
from tensorflow.gfile import Glob as glob

class Corpus(object):
  def __init__(self, path, dataset, *args, **kwargs):
    self.dataset = dataset
    self.vocab = Vocab(*args, **kwargs)

    train_path = self.vocab.count_file(s.path.join(path, "train.txt"))
    valid_path = self.vocab.count_file(s.path.join(path, "valid.txt"))
    test_path = self.vocab.count_file(s.path.join(path, "test.txt"))

    self.vocab.count_file(train_path)
    self.vocab.count_file(valid_path)
    self.vocab.count_file(test_path)
    self.vocab.build_vocab(add_bytes=True)

    self.train = train_path
    self.valid = self.vocab.encode_file(
        os.path.join(path, "valid.txt"), ordered=True, add_eos=False)
    self.test  = self.vocab.encode_file(
        os.path.join(path, "test.txt"), ordered=True, add_eos=False)
    self.cutoffs = []

  def convert_to_tfrecords(self, split, save_dir, bsz, tgt_len,
                           num_core_per_host, **kwargs):
    FLAGS = kwargs.get('FLAGS')

    file_names = []
    use_tpu = FLAGS.use_tpu and not (split == "test" and num_core_per_host == 1)

    record_name = "record_info-{}.bsz-{}.tlen-{}.json".format(
         split, bsz, tgt_len)
    record_info_path = os.path.join(save_dir, record_name)
    
    # pretty sure this is a tpu only thing
    bin_sizes = []

    if split == "train":
      np.random.seed(123456)
      num_batch = 0

      for shard in self.file_sharder(self.train, FLAGS.train_shard_size):
        num_shuffle = FLAGS.num_shuffle

        for shuffle in range(num_shuffle):
          print("Processing shard {} shuffle {}".format(shard, shuffle))
          basename = "train-{:03d}-{:02d}".format(shard, shuffle)
          np.random.shuffle(data_shard)
          file_name, num_batch_ = create_ordered_tfrecords(
              save_dir, basename, np.concatenate(data_shard), bsz, tgt_len,
              num_core_per_host,
              self.cutoffs, bin_sizes, use_tpu=use_tpu)
          file_names.append(file_name)
          num_batch += num_batch_

    else:
      file_name, num_batch = create_ordered_tfrecords(
          save_dir, split, getattr(self, split), bsz, tgt_len,
          num_core_per_host,
          self.cutoffs, bin_sizes, use_tpu=use_tpu)
      file_names.append(file_name)

    with open(record_info_path, "w") as fp:
      record_info = {
        "filenames": file_names,
        "bin_sizes": bin_sizes,
        "num_batch": num_batch
      }
      json.dump(record_info, fp)

  def file_sharder(self, file_name, shard_size):
      """ Shard a file into manageable sizes. """
      cur_shard_size = 0
      cur_shard = []

      with open(file_name, 'r') as f:
        for line in f:
            toks = self.vocab.tokenize(line)
            cur_shard.append(self.vocab.convert_to_nparray(toks))
            cur_shard_size += len(toks)

            if cur_shard_size >= shard_size:
                cur_shard = np.concatenate(cur_shard)
                print("Compiled shard of size {}".format(cur_shard_size))
                yield cur_shard
                
                cur_shard = []
                cur_shard_size = 0

        # want at least more than 50MB to write a shard
        if cur_shard_size >= 50000000:
            cur_shard = np.concatenate(cur_shard)
            yield cur_shard
        
def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _float_feature(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def create_ordered_tfrecords(save_dir, basename, data, batch_size, tgt_len,
                             num_core_per_host, cutoffs=[], bin_sizes=[], 
                             num_passes=1, use_tpu=False):

  if use_tpu:
    file_name = "{}.bsz-{}.tlen-{}.core-{}.tfrecords".format(
        basename, batch_size, tgt_len, num_core_per_host)
  else:
    file_name = "{}.bsz-{}.tlen-{}.tfrecords".format(
        basename, batch_size, tgt_len)

  save_path = os.path.join(save_dir, file_name)
  record_writer = tf.python_io.TFRecordWriter(save_path)

  batched_data = batchify(data, batch_size, num_passes)

  num_batch = 0
  # for t in range(0, batched_data.shape[1] - tgt_len - 1, tgt_len):
  for t in range(0, batched_data.shape[1] - 1, tgt_len):
    cur_tgt_len = min(batched_data.shape[1] - 1 - t, tgt_len)
    # drop the remainder if use tpu
    if use_tpu and cur_tgt_len < tgt_len: 
      break
    if num_batch % 500 == 0:
      print("  processing batch {}".format(num_batch))
    for idx in range(batch_size):
      inputs = batched_data[idx, t:t + cur_tgt_len]
      labels = batched_data[idx, t + 1:t + cur_tgt_len + 1]

      # features dict
      feature = {
          "inputs": _int64_feature(inputs),
          "labels": _int64_feature(labels),
      }

      if len(cutoffs) > 0 and use_tpu:
        # validate `bin_sizes` and `cutoffs`
        assert len(cutoffs) - len(bin_sizes) == 2, \
          "len(cutoffs) - len(bin_sizes) != 2"

        # mask for bin 0
        left, right = cutoffs[:2]
        inp_mask = ((inputs >= left) * (inputs < right)).astype(np.float32)
        tgt_mask = ((labels >= left) * (labels < right)).astype(np.float32)

        feature["inp_mask"] = _float_feature(inp_mask)
        feature["tgt_mask"] = _float_feature(tgt_mask)

        # refresh `inp_cnts` and `tgt_cnts` for each TPU core
        if idx % (batch_size // num_core_per_host) == 0:
          inp_cnts = [0] * len(bin_sizes)
          tgt_cnts = [0] * len(bin_sizes)

        head_labels = np.copy(labels)
        inp_pos_per_bin, tgt_pos_per_bin = [], []
        for b, (left, right) in enumerate(zip(cutoffs[1:-1], cutoffs[2:])):
          inp_pos = np.where((inputs >= left) * (inputs < right))[0]
          tgt_pos = np.where((labels >= left) * (labels < right))[0]
          inp_pos_per_bin.append(inp_pos)
          tgt_pos_per_bin.append(tgt_pos)

          head_labels[tgt_pos] = cutoffs[1] + b

        feature["head_labels"] = _int64_feature(head_labels)

        # permutation feature
        def _add_perm_feature(feature, pos_per_bin, cnts, prefix):
          for b, pos in enumerate(pos_per_bin):
            idx_tuple = []
            for p in pos:
              if cnts[b] < bin_sizes[b]:
                idx_tuple.append([p, cnts[b]])
                cnts[b] += 1
              else:
                break

            n_tup = len(idx_tuple)
            tup = np.array(idx_tuple).reshape(n_tup * 2)

            feature["{}_cnt_{}".format(prefix, b)] = _int64_feature([n_tup])
            feature["{}_tup_{}".format(prefix, b)] = _int64_feature(tup)

        _add_perm_feature(feature, inp_pos_per_bin, inp_cnts, "inp")
        _add_perm_feature(feature, tgt_pos_per_bin, tgt_cnts, "tgt")

      example = tf.train.Example(features=tf.train.Features(feature=feature))
      record_writer.write(example.SerializeToString())

    num_batch += 1

  record_writer.close()
  print("Done writing {}. batches: {}".format(file_name, num_batch))

  return file_name, num_batch


def get_lm_corpus(data_dir, dataset):
  fn = os.path.join(data_dir, "cache.pkl")
  print(fn)
  if exists(fn):
    print("Loading cached dataset...")
    with open(fn, "rb") as fp:
      corpus = pickle.load(fp)
  else:
    print("Producing dataset...")
    kwargs = {}
    if dataset in ["wt103", "wt2"]:
      kwargs["special"] = ["<eos>"]
      kwargs["lower_case"] = False
    elif dataset == "ptb":
      kwargs["special"] = ["<eos>"]
      kwargs["lower_case"] = True
    elif dataset == "lm1b":
      kwargs["special"] = []
      kwargs["lower_case"] = False
      kwargs["vocab_file"] = os.path.join(data_dir, "1b_word_vocab.txt")

    corpus = Corpus(data_dir, dataset, **kwargs)

    corpus_info = {
      "vocab_size" : len(corpus.vocab),
      "cutoffs" : corpus.cutoffs,
      "dataset" : corpus.dataset
    }
    with open(os.path.join(data_dir, "corpus-info.json"), "w") as fp:
      json.dump(corpus_info, fp)

  return corpus


def main(unused_argv):
  del unused_argv  # Unused

  corpus = get_lm_corpus(FLAGS.data_dir, FLAGS.dataset)

  save_dir = os.path.join(FLAGS.data_dir, "tfrecords")
  if not exists(save_dir):
    makedirs(save_dir)

  # # test mode
  if FLAGS.per_host_test_bsz > 0:
    corpus.convert_to_tfrecords("test", save_dir, FLAGS.per_host_test_bsz,
                                FLAGS.tgt_len, FLAGS.num_core_per_host, 
                                FLAGS=FLAGS)
    return

  for split, batch_size in zip(
      ["train", "valid"],
      [FLAGS.per_host_train_bsz, FLAGS.per_host_valid_bsz]):

    if batch_size <= 0: continue
    print("Converting {} set...".format(split))
    corpus.convert_to_tfrecords(split, save_dir, batch_size, FLAGS.tgt_len,
                                FLAGS.num_core_per_host, FLAGS=FLAGS)

    fn = os.path.join(FLAGS.data_dir, "cache.pkl")
    print("Saving dataset...")
    with open(fn, "wb") as fp:
      del corpus.valid
      del corpus.test
      pickle.dump(corpus, fp, protocol=2)

def load_record_info(record_info_dir, split, per_host_bsz, tgt_len,
                     num_core_per_host, use_tpu):
  if use_tpu:
    record_name = "record_info-{}.bsz-{}.tlen-{}.core-{}.json".format(
        split, per_host_bsz, tgt_len, num_core_per_host)
  else:
    record_name = "record_info-{}.bsz-{}.tlen-{}.json".format(
        split, per_host_bsz, tgt_len)

  record_info_path = os.path.join(record_info_dir, record_name)
  with open(record_info_path, "r") as fp:
    record_info = json.load(fp)

  return record_info

def get_input_fn(record_info_dir, split, per_host_bsz, tgt_len,
                 num_core_per_host, num_hosts=1, use_tpu=False):
  """Creates input function."""
  record_info = load_record_info(record_info_dir, split, per_host_bsz, tgt_len,
                                 num_core_per_host, use_tpu=use_tpu)

  file_names = record_info["filenames"]
  bin_sizes = record_info["bin_sizes"]
  num_batch = record_info["num_batch"]

  tf.logging.info("[{}] File names {}".format(split, file_names))

  def input_fn(params):
    # per-core batch size
    per_core_bsz = params["batch_size"]

    # data_dir could be a remote path, e.g., a google storage url
    data_dir = params["data_dir"]

    def parser(record):

      record_spec = {
          "inputs": tf.VarLenFeature(tf.int64),
          "labels": tf.VarLenFeature(tf.int64),
      }

      # retrieve serialized example
      example = tf.parse_single_example(
          serialized=record,
          features=record_spec)

      # cast int64 into int32
      # cast sparse to dense
      for key in list(example.keys()):
        val = example[key]
        if tf.keras.backend.is_sparse(val):
          val = tf.sparse.to_dense(val)
        if val.dtype == tf.int64:
          val = tf.to_int32(val)
        example[key] = val

      return example["inputs"], example["labels"]

    file_paths = []
    for file_name in file_names:
      file_path = os.path.join(data_dir, file_name)
      file_paths.append(file_path)

    if split == "train":
      dataset = tf.data.Dataset.from_tensor_slices(file_paths)
      if len(file_paths) > 1:
        dataset = dataset.shuffle(len(file_paths)).repeat()
        dataset = tf.data.TFRecordDataset(dataset)
      else:
        dataset = tf.data.TFRecordDataset(dataset)

      dataset = dataset.map(parser).cache().repeat()
      dataset = dataset.batch(per_core_bsz, drop_remainder=True)
      dataset = dataset.prefetch(num_core_per_host * per_core_bsz)
    else:
      # do not shuffle, repeat or cache in evaluation
      dataset = tf.data.Dataset.from_tensor_slices(file_paths)
      dataset = tf.data.TFRecordDataset(dataset)
      dataset = dataset.map(parser)
      dataset = dataset.batch(per_core_bsz, drop_remainder=True)

    return dataset

  if split == "train" and num_hosts > 1:
    record_info["num_batch"] = num_batch // num_hosts

  return input_fn, record_info

def get_corpus_info(corpus_info_path):
  with open(corpus_info_path, "r") as fp:
    corpus_info = json.load(fp)
  return corpus_info

if __name__ == "__main__":
  FLAGS = flags.FLAGS
  flags.DEFINE_string("data_dir", None,
        help="Location of the data corpus")
  flags.DEFINE_enum("dataset", "wt103",
        ["ptb", "wt2", "wt103", "lm1b", "enwik8", "text8"],
        help="Dataset name.")
  flags.DEFINE_integer("per_host_train_bsz", 60,
        help="train batch size each host")
  flags.DEFINE_integer("per_host_valid_bsz", 60,
        help="valid batch size each host")
  flags.DEFINE_integer("per_host_test_bsz", 0,
        help="If > 0, enter test mode and process test set only."
             "Otherwise, process train and dev sets only.")
  flags.DEFINE_integer("tgt_len", 70,
        help="number of tokens to predict")
  flags.DEFINE_integer("train_shard_size", 1000000000,
        help="How many bytes to keep in each shard. default 1GB")
  flags.DEFINE_integer("max_batch", -1,
        help="run in debug mode")
  flags.DEFINE_integer("num_core_per_host", 8,
        help="8 for TPU v2.")
  flags.DEFINE_bool("debug", default=False,
        help="Process only the first batch without shuffle for lm1b.")
  flags.DEFINE_integer("num_procs", 1,
        help="number of processes")
  flags.DEFINE_integer("num_passes", 10,
        help="number of passes when use_tpu=True")
  flags.DEFINE_integer("num_shuffle", 4,
        help="number of shuffles for lm1b")
  flags.DEFINE_bool("use_tpu", True,
        help="use tpu")

  tf.app.run(main)
