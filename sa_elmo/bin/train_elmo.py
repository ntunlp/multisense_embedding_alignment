
import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)

    # define the options
    batch_size = 256  # batch size for each GPU
    n_gpus = 2

    # number of tokens in training data (this for 1B Word Benchmark)
    # de
    # n_train_tokens = 432528096

    # en
    n_train_tokens = 768646526
    #n_train_tokens =  112869180
    #n_train_tokens =  23449557


    options = {
     'bidirectional': True,

     'char_cnn': {'activation': 'relu',
      'embedding': {'dim': 16},
      'filters': [[1, 32],
       [2, 32],
       [3, 64],
       [4, 128],
       [5, 256],
       [6, 512],
       [7, 1024]],
      'max_characters_per_token': 50,
      'n_characters': 261,
      'n_highway': 2},

     'dropout': 0.1,

     'lstm': {
      'cell_clip': 3,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': 512,
      'use_skip_connections': True},

     'all_clip_norm_val': 10.0,

     'n_epochs': 10,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 16,
     'n_negative_samples_batch': 8192,
     'n_context': 6,
     'cluster_proj_dim': 16,
     'pca_sample': 20000,
     # reset softmax weights at the specified batch
     #'reset_softmax_weight_batches': [2000, 22000],
     # record frequency of each output embedding vector selected when computing softmax
     'context_select_log': True,
     # length of window to record how many times each context is selected
     'context_record_window': 100,
     # must set context_select_log to True when setting remove_less_freqent_contexts > 0.
     'remove_less_freqent_contexts': 0.1,
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir, vocab=vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')

    args = parser.parse_args()
    main(args)

