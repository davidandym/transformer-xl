# coding=utf-8

""" Extract sentence features using a pretrained transformer-XL.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from absl import flags
import model
from data_utils import get_lm_corpus, get_corpus_info, Corpus

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string("model_checkpoint", default=None,
                    help="Points to a pretrained TXL checkpoint")
flags.DEFINE_string("data_dir", default=None,
                    help="Directory with Corpus info and object")
flags.DEFINE_string("sentences_file", default=None,
                    help="Sentences input file")
flags.DEFINE_string("sentence_reps_out", default=None,
                    help="Where to dump sentence representations")
flags.DEFINE_bool("backwards", default=False,
                  help=("Whether or not the LM is a backwards LM. (reverse input"))

# Model config
flags.DEFINE_integer("tgt_len", default=70,
                     help="Number of steps to predict")
flags.DEFINE_integer("mem_len", default=70,
                     help="Number of steps to cache")
flags.DEFINE_integer("clamp_len", default=-1,
                     help="Clamp length")
flags.DEFINE_bool("same_length", default=False,
                  help="Same length attention")

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
flags.DEFINE_enum("init", default="normal", enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("proj_init_std", default=0.01,
                   help="Initialization std for embedding projection.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

FLAGS = flags.FLAGS

def single_core_graph(n_token, cutoffs, is_training, inp, tgt, mems):
    """ Build the computation graph. """
    model_fn = get_model_fn(
        n_token=n_token,
        cutoffs=cutoffs)

    model_ret = model_fn(
        inp=inp,
        tgt=tgt,
        mems=mems,
        is_training=is_training)

    return model_ret


def get_model_fn(n_token, cutoffs):
    """ Builds up the computations graph, this is copied from the
    transformer-xl repo evaluation code, with a little modification to extract
    features.
    """
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

        if is_training:
            all_vars = tf.trainable_variables()
            grads = tf.gradients(loss, all_vars)
            grads_and_vars = list(zip(grads, all_vars))

            return loss, new_mems, grads_and_vars

        return loss, new_mems, outputs

    return model_fn


def load_dataset():
    """ Load the sentences, separate them into partitions of size tgt_len """
    sents = []
    part_size = FLAGS.tgt_len

    data_in_f = open(FLAGS.sentences_file, 'r')

    for line in data_in_f:
        line = line.strip()
        if line == "":
            continue
        symbols = line.split()

        if FLAGS.backwards:
            symbols = symbols[::-1]
        # annoying arithmetic
        num_sent_parts = len(symbols) // part_size if \
                len(symbols) % part_size == 0 else len(symbols) // part_size + 1

        # split sentence into partitions
        partitions = []
        cur = 0
        for _ in range(num_sent_parts):
            if cur+part_size > len(symbols):
                partitions.append(symbols[cur:])
            else:
                partitions.append(symbols[cur:cur+part_size])
            cur += part_size
        sents.append(partitions)

    return sents


def eval_input_fn(sents):
    """ Build up an input function to pass data into our comp graph.
    Takes in sentences, which have already been separated into paritions.
    """
    corpus = get_lm_corpus(FLAGS.data_dir, None)
    vocab = corpus.vocab

    def generator():
        for sent in sents:
            for partition in sent:
                ids = vocab.get_indices(partition)
                # the labels don't matter right now
                features = ids
                labels = ids
                yield features, labels

    # For now, just going to iterate one-by-one to manually manage the memory
    dataset = tf.data.Dataset.from_generator(generator, (tf.int32, tf.int32))
    dataset = dataset.batch(1, drop_remainder=False)
    return dataset


def main(unused_argv):
    """ Load sentences and pre-trained model, output representations. """
    del unused_argv  # Unused
    tf.logging.set_verbosity(tf.logging.INFO)

    corpus_info = get_corpus_info('{}/corpus-info.json'.format(FLAGS.data_dir))
    n_token = corpus_info["vocab_size"]
    print(n_token)
    cutoffs = corpus_info["cutoffs"][1:-1]

    sentences = load_dataset()
    eval_dataset = eval_input_fn(sentences)
    input_feed, label_feed = eval_dataset.make_one_shot_iterator().get_next()


    # Build the computations graph.
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
        saver.restore(sess, FLAGS.model_checkpoint)

        sentence_representations = []
        # iterate over sentences
        for sentence in sentences:
            char_reps_np = None
            tower_mems_np = \
                    [np.zeros([FLAGS.mem_len, 1, FLAGS.d_model], dtype=np.float32)
                     for layer in range(FLAGS.n_layer)]

            # iterate over paritions
            for _ in sentence:
                fetches = [loss, new_mem, outputs]
                feed_dict = {}
                for m_ref, m_np in zip(mems, tower_mems_np):
                    feed_dict[m_ref] = m_np

                # run the graph on our next input, store new memory and reps
                fetched = sess.run(fetches, feed_dict=feed_dict)
                _, tower_mems_np, char_rep = fetched[:3]

                # concat the partition back into the sentence
                char_rep = np.squeeze(char_rep, axis=1)
                if char_reps_np is None:
                    char_reps_np = char_rep
                else:
                    char_reps_np = np.concatenate((char_reps_np, char_rep), axis=0)
            
            if FLAGS.backwards:
                char_reps_np = np.flip(char_reps_np, axis=0)

            sentence_representations.append(char_reps_np)

    tf.logging.info("Extracted features for {} sentences.".format(len(sentence_representations)))
    tf.logging.info("Saving the representations here: {}".format(FLAGS.sentence_reps_out))
    np.save(FLAGS.sentence_reps_out, sentence_representations)

if __name__ == '__main__':
    tf.app.run()
