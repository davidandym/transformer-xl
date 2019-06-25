# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

from absl import flags
import numpy as np
import tensorflow as tf
import model
import data_utils

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string("model_dir",
                    default="pretrained_xl/tf_enwik8/model/model.ckpt-0",
      help="Estimator model_dir.")
flags.DEFINE_string("input_sents", default=None,
      help="File with sentences to extract features from")
flags.DEFINE_string("data_dir", default="pretrained_xl/tf_enwik9/data",
      help="Directory with Corpus info and object")

# Optimization config
flags.DEFINE_float("learning_rate", default=2.5e-4,
      help="Maximum learning rate.")
flags.DEFINE_float("clip", default=0.25,
      help="Gradient clipping value.")
# for cosine decay
flags.DEFINE_float("min_lr_ratio", default=0.004,
      help="Minimum ratio learning rate.")
flags.DEFINE_integer("warmup_steps", default=0,
      help="Number of steps for linear lr warmup.")

# Evaluation config
flags.DEFINE_integer("max_eval_batch", default=-1,
      help="Set -1 to turn off. Only used in test mode.")

# Model config
flags.DEFINE_integer("tgt_len", default=70,
      help="Number of steps to predict")
flags.DEFINE_integer("mem_len", default=70,
      help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
      help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")

flags.DEFINE_integer("n_layer", default=6,
      help="Number of layers.")
flags.DEFINE_integer("d_model", default=500,
      help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=500,
      help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=10,
      help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=50,
      help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=1000,
      help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.1,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
      help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
      help="untie r_w_bias and r_r_bias")

# Adaptive Softmax / Embedding
flags.DEFINE_bool("tie_weight", default=True,
      help="Tie embedding and softmax weight.")
flags.DEFINE_integer("div_val", default=1,
      help="Divide the embedding size by this val for each bin")
flags.DEFINE_bool("proj_share_all_but_first", default=False,
      help="True to share all but first projs, False not to share.")
flags.DEFINE_bool("proj_same_dim", default=True,
      help="Project the bin with the same dimension.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
      enum_values=["normal", "uniform"],
      help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
      help="Initialization std when init is normal.")
flags.DEFINE_float("proj_init_std", default=0.01,
      help="Initialization std for embedding projection.")
flags.DEFINE_float("init_range", default=0.1,
      help="Initialization std when init is uniform.")

FLAGS = flags.FLAGS

def single_core_graph(n_token, cutoffs, is_training, inp, tgt, mems):
    # build the computation graph for one core
    # n_token is vocab size
    model_fn = get_model_fn(
        n_token=n_token,
        cutoffs=cutoffs)

    # mems are just a placeholder, so that makes sense.
    model_ret = model_fn(
        inp=inp,
        tgt=tgt,
        mems=mems,
        is_training=is_training)

    return model_ret


def get_model_fn(n_token, cutoffs):
  def model_fn(inp, tgt, mems, is_training):
    inp = tf.transpose(inp, [1, 0])
    tgt = tf.transpose(tgt, [1, 0])

    initializer = tf.initializers.random_uniform(
        minval=-FLAGS.init_range,
        maxval=FLAGS.init_range,
        seed=None)
    proj_initializer = tf.initializers.random_normal(
        stddev=FLAGS.proj_init_std,
        seed=None)

    tie_projs = [False for _ in range(len(cutoffs) + 1)]
    if FLAGS.proj_share_all_but_first:
      for i in range(1, len(tie_projs)):
        tie_projs[i] = True

    loss, new_mems, outputs = model.transformer(
        dec_inp=inp,
        target=tgt,
        mems=mems,
        n_token=n_token,
        n_layer=FLAGS.n_layer,
        d_model=FLAGS.d_model,
        d_embed=FLAGS.d_embed,
        n_head=FLAGS.n_head,
        d_head=FLAGS.d_head,
        d_inner=FLAGS.d_inner,
        dropout=FLAGS.dropout,
        dropatt=FLAGS.dropatt,
        initializer=initializer,
        proj_initializer=proj_initializer,
        is_training=is_training,
        mem_len=FLAGS.mem_len,
        cutoffs=cutoffs,
        div_val=FLAGS.div_val,
        tie_projs=tie_projs,
        input_perms=None,
        target_perms=None,
        head_target=None,
        same_length=FLAGS.same_length,
        clamp_len=FLAGS.clamp_len,
        use_tpu=False,
        untie_r=FLAGS.untie_r,
        proj_same_dim=FLAGS.proj_same_dim,
        return_outputs=True)

    # number of parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('#params: {}'.format(num_params))

    format_str = '{{:<{0}s}}\t{{}}'.format(
        max([len(v.name) for v in tf.trainable_variables()]))
    for v in tf.trainable_variables():
      tf.logging.info(format_str.format(v.name, v.get_shape()))

    if is_training:
      all_vars = tf.trainable_variables()
      grads = tf.gradients(loss, all_vars)
      grads_and_vars = list(zip(grads, all_vars))

      return loss, new_mems, grads_and_vars
    else:
      return loss, new_mems, outputs

  return model_fn


def eval_input_fn():
    # their input function returns a dataset object.
    batch_size = 1

    corpus = data_utils.get_lm_corpus('pretrained_xl/tf_enwik8/data', None)
    vocab = corpus.vocab

    # test sentence
    test_sentence = u'Hi my name is David'
    test_bytes = map(ord, test_sentence.encode('utf-8'))
    test_bytes += [32 for _ in range(70-len(test_bytes))]
    test_bytes_input = [str(b) for b in test_bytes]
    test_sentence_ids = vocab.get_indices(test_bytes_input)

    # perhaps a fake parsed tf record
    example = {
        'inputs': test_sentence_ids,
        'labels': test_sentence_ids
    }

    def generator():
        features = example['inputs']
        labels = example['labels']
        for _ in range(10):
            yield features, labels
    
    dataset = tf.data.Dataset.from_generator(generator, (tf.int32, tf.int32))
    dataset = dataset.batch(1, drop_remainder=True)

    return dataset

def main(unused_argv):
    del unused_argv  # Unused
    tf.logging.set_verbosity(tf.logging.INFO)


    # load corpus shit
    corpus_info = \
            data_utils.get_corpus_info('pretrained_xl/tf_enwik8/data/corpus-info.json')
    n_token = corpus_info["vocab_size"]
    cutoffs = corpus_info["cutoffs"][1:-1]


    # create a dataset with the fake example
    eval_dataset = eval_input_fn()
    input_feed, label_feed = eval_dataset.make_one_shot_iterator().get_next()


    # build the comp graph for our cores... probably just one?
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):

        mems = [tf.placeholder(tf.float32, 
                               [FLAGS.mem_len, 1, FLAGS.d_model])
                for _ in range(FLAGS.n_layer)]
       
        loss, new_mem, outputs = single_core_graph(
            n_token=n_token,
            cutoffs=cutoffs,
            is_training=False,
            inp=input_feed,
            tgt=label_feed,
            mems=mems)

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        print(FLAGS.model_dir)
        # eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
        
        saver.restore(sess, FLAGS.model_dir)

        tower_mems_np = \
            [np.zeros([FLAGS.mem_len, 1, FLAGS.d_model], dtype=np.float32)
                for layer in range(FLAGS.n_layer)]

        fetches = [loss, new_mem, outputs]

        feed_dict = {}
        for m, m_np in zip(mems, tower_mems_np):
            feed_dict[m] = m_np

        fetched = sess.run(fetches, feed_dict=feed_dict)

        tf.logging.info("outputs: {}".format(fetched[2]))

if __name__ == '__main__':
    tf.app.run()
