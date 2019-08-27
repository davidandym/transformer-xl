from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import csv

from absl import flags
import absl.logging as _logging

import tensorflow as tf
import model
import data_utils
from data_utils import Corpus
from gpu_utils import assign_to_gpu, average_grads_and_vars

import numpy as np

def make_flags():
    flags.DEFINE_integer("num_gpu", default=8,
          help="Number of cores per host")
    
    # Directory paths
    flags.DEFINE_string("data_dir", default="",
          help="Path to tf-records directory.")
    flags.DEFINE_string("record_info_dir", default="",
          help="Path to local directory containing filenames.txt.")
    flags.DEFINE_string("corpus_info_path", default="",
          help="Path to corpus-info.json file.")
    flags.DEFINE_string("model_dir", default=None,
          help="Estimator model_dir.")
    flags.DEFINE_string("log_file", default='ext-logs.csv',
          help="Log file. Will be stored in model_dir")
    flags.DEFINE_string("mode", default='train',
          help="choices eval, train")
    
    # Optimization config
    flags.DEFINE_float("learning_rate", default=2.5e-4,
          help="Maximum learning rate.")
    flags.DEFINE_float("clip", default=0.25,
          help="Gradient clipping value.")
    flags.DEFINE_float("min_lr_ratio", default=0.004,
          help="Minimum ratio learning rate.")
    flags.DEFINE_integer("warmup_steps", default=0,
          help="Number of steps for linear lr warmup.")
    
    # Training config
    flags.DEFINE_integer("train_batch_size", default=60,
          help="Size of train batch.")
    flags.DEFINE_integer("eval_batch_size", default=60,
          help="Size of valid batch.")
    flags.DEFINE_integer("epochs", default=10,
          help="Total number of training steps.")
    flags.DEFINE_integer("iterations", default=500,
          help="Number of iterations per repeat loop.")
    flags.DEFINE_integer("save_steps", default=10000,
          help="number of steps for model checkpointing.")
    
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
    
    flags.DEFINE_float("init_std", default=0.02,
          help="Initialization std when init is normal.")
    flags.DEFINE_float("proj_init_std", default=0.01,
          help="Initialization std for embedding projection.")
    flags.DEFINE_float("init_range", default=0.1,
          help="Initialization std when init is uniform.")

make_flags()
FLAGS = flags.FLAGS


def get_model_fn(n_token, cutoffs):
    def model_fn(inp, tgt, mems, is_training):
        inp = tf.transpose(inp, [1, 0])
        tgt = tf.transpose(tgt, [1, 0])


        initializer = tf.initializers.random_normal(
            stddev=FLAGS.init_std,
            seed=None)
        proj_initializer = tf.initializers.random_normal(
            stddev=FLAGS.proj_init_std,
            seed=None)

        tie_projs = [False for _ in range(len(cutoffs) + 1)]
        if FLAGS.proj_share_all_but_first:
            for i in range(1, len(tie_projs)):
                tie_projs[i] = True

        loss, new_mems = model.transformer(
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
            proj_same_dim=FLAGS.proj_same_dim)

        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        if is_training:
            all_vars = tf.trainable_variables()
            grads = tf.gradients(loss, all_vars)
            grads_and_vars = list(zip(grads, all_vars))
            return loss, new_mems, grads_and_vars
        else:
            return loss, new_mems

    return model_fn


def single_core_graph(n_token, cutoffs, is_training, inp, tft, mems):
    model_fn = get_model_fn(
        n_token=n_token,
        cutoffs=cutoffs)

    model_ret = model_fn(
        inp=inp,
        tgt=tgt,
        mems=mems,
        is_training=is_training)

    return model_ret


def train_epoch(epoch, csv_logger, n_token, cutoffs):
    ps_device = "/gpu:0"

    train_input_fn, train_record_info = data_utils.get_input_fn(
        record_info_dir=FLAGS.record_info_dir,
        split="train",
        per_host_bsz=FLAGS.train_batch_size,
        tgt_len=FLAGS.tgt_len,
        num_core_per_host=FLAGS.num_gpu,
        num_hosts=1,
        use_tpu=False)

    tf.logging.info("-" * 30)
    tf.logging.info("Starting epoch {}!".format(epoch))
    tf.logging.info("num of batches {}".format(train_record_info["num_batch"]))
    num_batch = train_record_info["num_batch"]

    train_set = train_input_fn({
        "batch_size": FLAGS.train_batch_size,
        "data_dir": FLAGS.data_dir})

    input_feed, label_feed = train_set.make_one_shot_iterator().get_next()

    inputs = tf.split(input_feed, FLAGS.num_gpu, 0)
    labels = tf.split(label_feed, FLAGS.num_gpu, 0)

    per_core_bsz = FLAGS.train_batch_size // FLAGS.num_gpu
    tower_mems, tower_losses, tower_new_mems, tower_grads_and_vars = [], [], [], []

    for i in range(FLAGS.num_gpu):
        reuse = True if i > 0 else None
        with tf.device(assign_to_gpu(i, ps_device)), \
                tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

            mems_i = [tf.placeholder(tf.float32,
                                     [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                      for _ in range(FLAGS.n_layer)]

            loss_i, new_mems_i, grads_and_vars_i = single_core_graph(
                n_token=n_token,
                cutoffs=cutoffs,
                is_training=True,
                inp=inputs[i],
                tft=labels[i],
                mems=mems_i)

            tower_mems.append(mems_i)
            tower_losses.append(loss_i)
            tower_new_mems.append(new_mems_i)
            tower_grads_and_vars.append(grads_and_vars_i)

    if len(tower_losses) > 1:
        loss = tf.add_n(tower_losses) / len(tower_losses)
        grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
    else:
        loss = tower_losses[0]
        grads_and_vars = tower_grads_and_vars[0]

    grads, all_vars = zip(*grads_and_vars)

    clipped, gnorm = tf.clip_by_global_norm(grads, FLAGS.clip)
    grads_and_vars = list(zip(clipped, all_vars))

    global_step = tf.train.get_or_create_global_step()

    if FLAGS.warmup_steps > 0:
        warmup_lr = tf.to_float(global_step) / tf.to_float(FLAGS.warmup_steps) \
                    * FLAGS.learning_rate
    else:
        warmup_lr = 0.0

    decay_lr = tf.train.cosine_decay(
        FLAGS.learning_rate,
        global_step=global_step-FLAGS.warmup_steps,
        decay_steps=FLAGS.train_steps-FLAGS.warmup_steps,
        alpha=FLAGS.min_lr_ratio)


    learning_rate = tf.where(global_step < FLAGS.warmup_steps,
                             warmup_lr, decay_lr)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    tower_mems_np = [
        [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
            for layer in range(FLAGS.n_layer)]
        for core in range(FLAGS.num_gpu)
    ]

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        latest_ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
        if latest_ckpt is not None:
            tf.logging.info("loading saved model from {}".format(latest_ckpt))
            saver.restore(sess, latest_ckpt)
        else:
            tf.logging.info("No previously saved model. Starting from scratch!")

    fetches = [loss, tower_new_mems, global_step, gnorm, learning_rate, train_op]
    total_loss, prev_step = 0., -1

    for ba in range(num_batch):
        feed_dict = {}
        for i in range(FLAGS.num_core_per_host):
            for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
                feed_dict[m] = m_np
        fetched = sess.run(fetches, feed_dict=feed_dict)
        loss_np, tower_mems_np, curr_step = fetched[:3]
        total_loss += loss_np
        if curr_step > 0 and curr_step % FLAGS.iterations == 0:
            cur_loss = total_loss / (curr_step - prev_step)
            tf.logging.info("[{}] | gnorm {:.2f} lr {:8.6f} "
                "| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
                curr_step, fetched[-3], fetched[-2],
                curr_loss, math.exp(curr_loss), curr_loss / math.log(2)))
            log_dict = {
                'train_loss': curr_loss,
                'train_ppl': math.exp(curr_loss),
                'train_bpc': curr_loss / math.log(2),
                'lr': fetched[-2],
                'global_step': curr_step,
                'epoch': epoch
            }
            csv_logger.writerow(log_dict)
            total_loss, prev_step = 0., curr_step

        if curr_step > 0 and curr_step % FLAGS.save_steps == 0:
            save_path = os.path.join(FLAGS.model_dir, "model.ckpt")
            saver.save(sess, save_path)
            tf.logging.info("Finished Step : {}".format(curr_step))
            tf.logging.info("Model saved in path: {}".format(save_path))


    cur_loss = total_loss / (curr_step - prev_step)
    tf.logging.info("[{}] | gnorm {:.2f} lr {:8.6f} "
        "| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
        curr_step, fetched[-3], fetched[-2],
        curr_loss, math.exp(curr_loss), curr_loss / math.log(2)))

    save_path = os.path.join(FLAGS.model_dir, "model.ckpt")
    saver.save(sess, save_path)
    tf.logging.info("Finished Epoch {}".format(curr_step))
    tf.logging.info("Model saved in path: {}".format(save_path))
    tf.logging.info("-" * 30)
        

def evaluate(n_token, cutoffs, ps_device):
    ##### Get input function and model function

    ps_device = "/gpu:0"

    # eval input function returns a dataset obj.
    eval_input_fn, eval_record_info = data_utils.get_input_fn(
        record_info_dir=FLAGS.record_info_dir,
        split="valid", # train or valid
        per_host_bsz=FLAGS.eval_batch_size,
        tgt_len=FLAGS.tgt_len,
        num_core_per_host=FLAGS.num_gpu,
        num_hosts=1,
        use_tpu=False)

    num_batch = eval_record_info["num_batch"]
    if FLAGS.max_eval_batch > 0:
        num_batch = FLAGS.max_eval_batch
    tf.logging.info("num of batches {}".format(num_batch))

    ##### Create computational graph

    # this is a dataset obj.
    eval_set = eval_input_fn({
        "batch_size": FLAGS.eval_batch_size,
        "data_dir": FLAGS.data_dir})

    # gets the two feeds... We can simulate this with a generator.
    input_feed, label_feed = eval_set.make_one_shot_iterator().get_next()

    inputs = tf.split(input_feed, FLAGS.num_gpu, 0)
    labels = tf.split(label_feed, FLAGS., 0)

    per_core_bsz = FLAGS.eval_batch_size // FLAGS.num_core_per_host
    tower_mems, tower_losses, tower_new_mems = [], [], []

    for i in range(FLAGS.num_core_per_host):
        with tf.device(assign_to_gpu(i, ps_device)), \
            tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):

            mems_i = [tf.placeholder(tf.float32,
                          [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                      for _ in range(FLAGS.n_layer)]

            loss_i, new_mems_i = single_core_graph(
                n_token=n_token,
                cutoffs=cutoffs,
                is_training=False,
                inp=inputs[i],
                tgt=labels[i],
                mems=mems_i)

            tower_mems.append(mems_i)
            tower_losses.append(loss_i)
            tower_new_mems.append(new_mems_i)

    ## sum losses across towers
    if len(tower_losses) > 1:
      loss = tf.add_n(tower_losses) / len(tower_losses)
    else:
      loss = tower_losses[0]

    ##### Evaluation loop
    tower_mems_np = [
        [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
            for layer in range(FLAGS.n_layer)]
        for core in range(FLAGS.num_gpu)
    ]

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
        tf.logging.info("Evaluating {}".format(eval_ckpt_path))
        saver.restore(sess, eval_ckpt_path)

        fetches = [loss, tower_new_mems, tf.size(label_feed)]

        format_str = "  >> processing batch {{:{0}d}}/{{:{0}d}} ..".format(
            len(str(num_batch)))

        total_loss, total_cnt = 0, 0
        for step in range(num_batch):
            if step % (num_batch // 10) == 0:
                tf.logging.info(format_str.format(step, num_batch))

            feed_dict = {}
            for i in range(FLAGS.num_core_per_host):
                for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
                    feed_dict[m] = m_np

            fetched = sess.run(fetches, feed_dict=feed_dict)

            loss_np, tower_mems_np, cnt_np = fetched[:3]
            total_loss += loss_np * cnt_np
            total_cnt += cnt_np

    avg_loss = total_loss / total_cnt
    tf.logging.info("| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
        avg_loss, math.exp(avg_loss), avg_loss / math.log(2)))

    log_dict = {
        'valid_loss': avg_loss,
        'valid_ppl': math.exp(avg_loss),
        'valid_bpc': avg_loss / math.log(2)
    }
    return log_dict


def get_csv_logger():
    csv_file = open(FLAGS.log_file, 'w', newline='')
    fieldnames = ['train_loss', 'train_ppl', 'train_bpc', 'lr', 'valid_loss', \
                  'valid_ppl', 'valid_bpc', 'global_step', 'epoch']
    csv_logger = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_logger.writeheader()
    return csv_logger, csv_file


def main(unused_argv):
    del unused_argv

    tf.logging.set_verbosity(tf.logging.INFO)

    csv_logger, csv_file = get_csv_logger()
    corpus_info = data_utils.get_corpus_info(FLAGS.corpus_info_path)
    n_token = corpus_info["vocab_size"]
    cutoffs = corpus_info["cutoffs"][1:-1]
    tf.logging.info("n_token: {}".format(n_token))

    min_valid_loss = np.INF

    for epoch in range(FLAGS.train_epochs):
        train_epoch(epoch, csv_logger, n_token, cutoffs)

        tf.logging.info("Evaluating epoch {}".format(epoch))
        valid_log_dict = evaluate(n_token, cutoffs)
        valid_log_dict['epoch'] = epoch
        csv_logger.writerow(valid_log_dict)
        
        if valid_log_dict['valid_loss'] < min_valid_loss:
            min_valid_loss = valid_log_dict['valid_loss']
            tf.logging.info("New min loss {}".format(min_valid_loss))
            save_path = os.path.join(FLAGS.model_dir, "best_model.ckpt")
            tf.logging.info("Saving model in {}".format(save_path))
            saver.save(sess, save_path)
            
    csv_file.close()

if __name__ == "__main__":
  tf.app.run()
