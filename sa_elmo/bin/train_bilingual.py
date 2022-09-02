
import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab, load_dictionary
from bilm.data import BidirectionalLMDatasetBilingual


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)
    dictionary = load_dictionary(args.dictionary, vocab, 4)

    # define the options
    batch_size = 256  # batch size for each GPU
    n_gpus = 2

    # number of tokens in training data, l1 is the one with larger data size
    # n_train_tokens_l1 and n_train_tokens_l2 are used to compute the total number of tokens
    
    # en_de sample5 451353468
    n_train_tokens_l1 = 451353468
    n_train_tokens_l2 = 451353468
    
    # en_de sample
    #n_train_tokens_l1 = 19118892
    #n_train_tokens_l2 = 17165745
    #Training data sample ratio
    l1_sample_ratio = 0.5
    n_train_tokens = max(n_train_tokens_l1, n_train_tokens_l2) * 2


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
     'unroll_steps': 12,
     'n_negative_samples_batch': 8192,
     'n_context': 4,
     'cluster_proj_dim': 16,
     'pca_sample': 20000,
     # record frequency of each output embedding vector selected when computing softmax
     'context_select_log': True,
     # length of window to record how many times each context is selected
     'context_record_window': 100,
     # must set context_select_log to True when setting remove_less_freqent_contexts > 0.
     'remove_less_freqent_contexts': 0.1,
     # bilingual model
     'bilingual': True,
     'dictionary_shape': dictionary.shape
    }

    prefix1 = args.train_prefix1
    prefix2 = args.train_prefix2
    data = BidirectionalLMDatasetBilingual(prefix1, prefix2, vocab,
                                      l1_sample_ratio, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir, dictionary=dictionary, vocab=vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix1', help='Prefix for train files of language 1 (larger data size)')
    parser.add_argument('--train_prefix2', help='Prefix for train files of language 2')
    parser.add_argument('--dictionary', help='Dictionary file')

    args = parser.parse_args()
    main(args)

