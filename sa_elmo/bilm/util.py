import tensorflow as tf


def _sum_rows(x):
  """Returns a vector summing up each row of the matrix x."""
  # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
  # a matrix.  The gradient of _sum_rows(x) is more efficient than
  # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
  # we use _sum_rows(x) in the nce_loss() computation since the loss
  # is mostly used for training.
  cols = tf.shape(x)[1]
  ones_shape = tf.stack([cols, 1])
  ones = tf.ones(ones_shape, x.dtype)
  return tf.reshape(tf.matmul(x, ones), [-1])


def _cos_similarity_proj(cluster_proj, inputs, labels_w, labels_len):
  # project to a lower dimension for clustering, sum score for n projectors
  # cluster_proj shape [dim_orig, dim_proj]
  dim_proj = tf.shape(cluster_proj)[1]
  inputs_len = tf.shape(inputs)[0]
  inputs_proj = tf.matmul(inputs, cluster_proj)
  labels_w_proj = tf.matmul(labels_w, cluster_proj)
  row_wise_dots_proj = tf.math.multiply(
    tf.expand_dims(inputs_proj, 1),
    tf.reshape(labels_w_proj, [-1, labels_len, dim_proj]))
  dots_as_matrix_proj = tf.reshape(row_wise_dots_proj, [-1, dim_proj])
  labels_logits_proj = tf.reshape(
        tf.reduce_sum(dots_as_matrix_proj, 1),
        [-1, labels_len])
  labels_w_proj_norm = tf.reshape(
        tf.norm(tf.reshape(labels_w_proj, [-1, dim_proj]), axis=1),
        [-1, labels_len]) + 1e-6
  inputs_proj_norm = tf.reshape(
        tf.norm(tf.reshape(inputs_proj, [-1, dim_proj]), axis=1),
        [inputs_len, -1]) + 1e-6
  similarity_proj = (labels_logits_proj / labels_w_proj_norm) / inputs_proj_norm
  return similarity_proj


def _get_true_w_dist_loss(true_w, true_logits_onehot):
  true_w_selected = tf.expand_dims(true_logits_onehot, 2) * true_w
  true_w_selected = tf.reduce_sum(true_w_selected, 1)
  dist = true_w_selected - tf.reduce_mean(true_w, 1)
  dist = tf.math.sqrt(tf.reduce_sum(dist * dist, 1))
  return -tf.reduce_mean(dist)


def _remove_accidental_hits_from_sampled(labels,
                                         sampled,
                                         sampled_logits,
                                         num_sampled,
                                         num_true,
                                         ):
  acc_hits = tf.nn.compute_accidental_hits(
      labels, sampled, num_true=num_true)
  acc_indices, acc_ids, acc_weights = acc_hits

  # This is how SparseToDense expects the indices.
  acc_indices_2d = tf.reshape(acc_indices, [-1, 1])
  acc_ids_2d_int32 = tf.reshape(
      tf.dtypes.cast(acc_ids, tf.dtypes.int32), [-1, 1])
  sparse_indices = tf.concat([acc_indices_2d, acc_ids_2d_int32], 1,
                                    "sparse_indices")
  # Create sampled_logits_shape = [batch_size, num_sampled]
  sampled_logits_shape = tf.concat(
      [tf.shape(labels)[:1],
       tf.expand_dims(num_sampled, 0)], 0)
  if sampled_logits.dtype != acc_weights.dtype:
    acc_weights = tf.dtypes.cast(acc_weights, sampled_logits.dtype)
  sampled_logits += tf.sparse_to_dense(
      sparse_indices,
      sampled_logits_shape,
      acc_weights,
      default_value=0.0,
      validate_indices=False)
  return sampled_logits


def _remove_disabled_logits(valid_index,
                            labels,
                            label_logits):
  """add -inf to label_logits if corresponding index in labels is not in valid_index
       return the computed logits and mask. 1 in mask indicate a valid value, 0 indicates removed value
  """
  acc_hits = tf.nn.compute_accidental_hits(
        tf.dtypes.cast(tf.reshape(labels, [-1, 1]), tf.dtypes.int64),
        tf.dtypes.cast(tf.reshape(valid_index, [-1]), tf.dtypes.int64),
        num_true=1)
  acc_indices, acc_ids, acc_weights = acc_hits

  sampled_logits_shape = tf.cast(tf.shape(tf.reshape(label_logits, [-1])), tf.dtypes.int32)
  default_weight = tf.reshape(acc_weights, [-1])[0]
  zero_weights = tf.zeros_like(acc_weights)
  if label_logits.dtype != zero_weights.dtype:
    zero_weights = tf.dtypes.cast(zero_weights, label_logits.dtype)
  mask = tf.sparse_to_dense(
        acc_indices,
        sampled_logits_shape,
        zero_weights,
        default_value=1.0,
        validate_indices=False)
  mask = tf.reshape(mask, tf.shape(label_logits))
  label_logits += mask * default_weight
  return label_logits, tf.math.abs(mask - 1)


def _get_labels_dict_logits(labels_dict,
                            weights,
                            biases,
                            partition_strategy,
                            inputs,
                            remove_accidental_hits,
                            sampled,
                            sampled_logits,
                            num_sampled,
                            context_enabled=None,
                            valid_sample_index=None,
                            cluster_proj=None,
                            softmax_sense_all=None,
                            tk_context=None):
  if labels_dict.dtype != tf.dtypes.int64:
    labels_dict = tf.dtypes.cast(labels_dict, tf.dtypes.int64)
  labels_dict_len = labels_dict.get_shape().as_list()[-1]
  labels_dict_flat = tf.reshape(labels_dict, [-1])
  # trick here to change negative index (invalid translation) in labels_dict_flat to 0
  # the corresponding logits value will be set to -inf since the negative index is not in valid_sample_index
  labels_dict_flat_pos = tf.dtypes.cast(tf.nn.relu(labels_dict_flat), tf.dtypes.int64)
  dict_sense = tf.nn.embedding_lookup(
      softmax_sense_all, labels_dict_flat_pos, partition_strategy=partition_strategy)
  labels_dict_w = tf.nn.embedding_lookup(
      weights, labels_dict_flat_pos, partition_strategy=partition_strategy)
  labels_dict_b = tf.nn.embedding_lookup(
      biases, labels_dict_flat_pos, partition_strategy=partition_strategy)

  if labels_dict_w.dtype != inputs.dtype:
    labels_dict_w = tf.dtypes.cast(labels_dict_w, inputs.dtype)
  if labels_dict_b.dtype != inputs.dtype:
    labels_dict_b = tf.dtypes.cast(labels_dict_b, inputs.dtype)
  dim = tf.shape(labels_dict_w)[1]
  # dict_row_wise_dots dim [batch_size, labels_dict_len, dim]
  dict_row_wise_dots = tf.math.multiply(
    tf.expand_dims(inputs, 1),
    tf.reshape(labels_dict_w, [-1, labels_dict_len, dim]))
  dict_dots_as_matrix = tf.reshape(dict_row_wise_dots, [-1, dim])
  labels_dict_logits = tf.reshape(_sum_rows(dict_dots_as_matrix), [-1, labels_dict_len])
  labels_dict_b = tf.reshape(labels_dict_b, [-1, labels_dict_len])
  labels_dict_logits += labels_dict_b

  # project to a lower dimension for clustering
  labels_dict_cos_similarity = _cos_similarity_proj(cluster_proj, tk_context, dict_sense, labels_dict_len)

  if context_enabled is not None:
    labels_dict_logits, remove_mask = _remove_disabled_logits(valid_sample_index,
                                          labels_dict,
                                          labels_dict_logits)
    labels_dict_cos_similarity, remove_mask = _remove_disabled_logits(valid_sample_index,
                                          labels_dict,
                                          labels_dict_cos_similarity)


  labels_dict_logits_onehot = tf.one_hot(tf.math.argmax(labels_dict_cos_similarity, axis=1),
                                    labels_dict_len)
  # multiply the onehot matrix with remove_mask, in case there is no valid index in a row
  labels_dict_logits_onehot = tf.math.multiply(remove_mask, labels_dict_logits_onehot)
  if labels_dict_logits_onehot.dtype != labels_dict_logits.dtype:
      labels_dict_logits_onehot = tf.dtypes.cast(labels_dict_logits_onehot,
                                      labels_dict_logits.dtype)

  ### remove unselected labels
  ### labels_dict: may become 0 if there is no valid translation, but does not affect result
  ##labels_dict = tf.dtypes.cast(
  ##      tf.reduce_sum(labels_dict * tf.dtypes.cast(labels_dict_logits_onehot, labels_dict.dtype), 1, keepdims=True),
  ##      labels_dict.dtype)
  ##labels_dict_logits = tf.dtypes.cast(
  ##      tf.reduce_sum(labels_dict_logits * labels_dict_logits_onehot, 1, keepdims=True),
  ##      labels_dict_logits.dtype)
  ##labels_dict_logits_onehot = tf.dtypes.cast(
  ##      tf.reduce_sum(labels_dict_logits_onehot, 1, keepdims=True),
  ##      labels_dict_logits_onehot.dtype)

  if remove_accidental_hits:
    sampled_logits = _remove_accidental_hits_from_sampled(
                                         labels_dict,
                                         sampled,
                                         sampled_logits,
                                         num_sampled,
                                         labels_dict_len,
                                         )

  return labels_dict_logits, labels_dict_logits_onehot, sampled_logits


def sampled_softmax_loss_multi_context(weights,
                         biases,
                         labels,
                         inputs,
                         num_sampled,
                         num_classes,
                         num_context=1,
                         sampled_values=None,
                         remove_accidental_hits=True,
                         partition_strategy="mod",
                         name="sampled_softmax_loss",
                         seed=None,
                         context_enabled=None,
                         dictionary=None,
                         cluster_proj=None,
                         softmax_sense_all=None,
                         tk_context=None,
                         ):

  # get corresponding embedding indices for each label
  labels_context = tf.concat([labels * num_context + n for n in range(num_context)], 1)
  tk_context = tf.stop_gradient(tk_context, name='tk_context_stop_gradient')
  if dictionary is not None:
    labels_dict = tf.nn.embedding_lookup(
        dictionary, labels, partition_strategy=partition_strategy)
    # reshape dictionary from [batch_size, 1, dict_size] to  [batch_size, dict_size]
    labels_dict = tf.reshape(labels_dict, [-1, tf.shape(labels_dict)[-1]])
    # convert vocatulary index to the corresponding embedding index
    # size [batch_size, dict_size * num_context]
    labels_dict = tf.concat([labels_dict * num_context + n for n in range(num_context)], 1)
    labels_dict = tf.stop_gradient(labels_dict, name="labels_dict_stop_gradient")
  else:
    labels_dict = None

  # labels_selected: context id selected
  logits, labels_context, labels_selected, true_w_dist_loss = compute_sampled_logits_multi_context(
      weights=weights,
      biases=biases,
      labels=labels_context,
      inputs=inputs,
      num_sampled=num_sampled,
      num_classes=num_classes,
      num_true=num_context,
      sampled_values=sampled_values,
      subtract_log_q=False,
      remove_accidental_hits=remove_accidental_hits,
      partition_strategy=partition_strategy,
      name=name,
      seed=seed,
      context_enabled=context_enabled,
      labels_dict=labels_dict,
      cluster_proj=cluster_proj,
      softmax_sense_all=softmax_sense_all,
      tk_context=tk_context,
      )
  labels_context = tf.stop_gradient(labels_context, name="labels_stop_gradient")
  sampled_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=labels_context, logits=logits)
  # sampled_losses is a [batch_size] tensor.
  return sampled_losses, labels_selected, true_w_dist_loss


def compute_sampled_logits_multi_context(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            sampled_values=None,
                            subtract_log_q=True,
                            remove_accidental_hits=False,
                            partition_strategy="mod",
                            name=None,
                            seed=None,
                            context_enabled=None,
                            labels_dict=None,
                            cluster_proj=None,
                            softmax_sense_all=None,
                            tk_context=None,
                            ):

  if not isinstance(weights, list):
    weights = [weights]
  with tf.name_scope(name, "compute_sampled_logits",
                      weights + [biases, inputs, labels]):
    if labels.dtype != tf.dtypes.int64:
      labels = tf.dtypes.cast(labels, tf.dtypes.int64)
    labels_flat = tf.reshape(labels, [-1])
    softmax_sense = tf.nn.embedding_lookup(softmax_sense_all, labels_flat,
                                           partition_strategy='mod')

    if context_enabled is not None:
      valid_sample_index = tf.where(tf.greater(context_enabled, 1e-6))
      valid_sample_index = tf.reshape(valid_sample_index, [-1])
      sampled = tf.random.shuffle(valid_sample_index)[:num_sampled]
      valid_sample_index = tf.stop_gradient(valid_sample_index)
      sampled = tf.stop_gradient(sampled)
    else:
      # Sample the negative labels.
      #   sampled shape: [num_sampled] tensor
      #   true_expected_count shape = [batch_size, 1] tensor
      #   sampled_expected_count shape = [num_sampled] tensor
      if sampled_values is None:
        sampled_values = tf.random.uniform_candidate_sampler(
          true_classes=labels,
          num_true=num_true,
          num_sampled=num_sampled,
          unique=True,
          range_max=num_classes,
          seed=seed)

      # NOTE: pylint cannot tell that 'sampled_values' is a sequence
      # pylint: disable=unpacking-non-sequence
      sampled, true_expected_count, sampled_expected_count = (
          tf.stop_gradient(s) for s in sampled_values)


    # pylint: enable=unpacking-non-sequence
    sampled = tf.dtypes.cast(sampled, tf.dtypes.int64)

    # labels_flat is a [batch_size * num_true] tensor
    # sampled is a [num_sampled] int tensor
    all_ids = tf.concat([labels_flat, sampled], 0)

    # Retrieve the true weights and the logits of the sampled weights.

    # weights shape is [num_classes, dim]
    all_w = tf.nn.embedding_lookup(
        weights, all_ids, partition_strategy=partition_strategy)
    if all_w.dtype != inputs.dtype:
      all_w = tf.dtypes.cast(all_w, inputs.dtype)

    # true_w shape is [batch_size * num_true, dim]
    true_w = tf.slice(all_w, [0, 0],
                             tf.stack([tf.shape(labels_flat)[0], -1]))

    sampled_w = tf.slice(
        all_w, tf.stack([tf.shape(labels_flat)[0], 0]), [-1, -1])
    # inputs has shape [batch_size, dim]
    # sampled_w has shape [num_sampled, dim]
    # Apply X*W', which yields [batch_size, num_sampled]
    sampled_logits = tf.matmul(inputs, sampled_w, transpose_b=True)

    # Retrieve the true and sampled biases, compute the true logits, and
    # add the biases to the true and sampled logits.
    all_b = tf.nn.embedding_lookup(
        biases, all_ids, partition_strategy=partition_strategy)
    if all_b.dtype != inputs.dtype:
      all_b = tf.dtypes.cast(all_b, inputs.dtype)
    # true_b is a [batch_size * num_true] tensor
    # sampled_b is a [num_sampled] float tensor
    true_b = tf.slice(all_b, [0], tf.shape(labels_flat))
    sampled_b = tf.slice(all_b, tf.shape(labels_flat), [-1])

    # inputs shape is [batch_size, dim]
    # true_w shape is [batch_size * num_true, dim]
    # row_wise_dots is [batch_size, num_true, dim]
    dim = tf.shape(true_w)[1:2]
    new_true_w_shape = tf.concat([[-1, num_true], dim], 0)
    row_wise_dots = tf.math.multiply(
        tf.expand_dims(inputs, 1),
        tf.reshape(true_w, new_true_w_shape))
    # We want the row-wise dot plus biases which yields a
    # [batch_size, num_true] tensor of true_logits.
    dots_as_matrix = tf.reshape(row_wise_dots,
                                       tf.concat([[-1], dim], 0))
    true_logits = tf.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
    true_b = tf.reshape(true_b, [-1, num_true])
    true_logits += true_b
    sampled_logits += sampled_b

    # project to a lower dimension for clustering
    # true_w_cos_similarity is used to select the closest true_w vector
    true_w_cos_similarity = _cos_similarity_proj(cluster_proj, tk_context, softmax_sense, num_true)

    if context_enabled is not None:
      true_logits, remove_mask = _remove_disabled_logits(valid_sample_index,
                                            labels,
                                            true_logits)
      true_w_cos_similarity, remove_mask = _remove_disabled_logits(valid_sample_index,
                                            labels,
                                            true_w_cos_similarity)

    # true_logits is a float tensor, ones_like(true_logits) is a float
    # tensor of ones. We then divide by num_true to ensure the per-example
    # labels sum to 1.0, i.e. form a proper probability distribution.
    true_logits_onehot = tf.one_hot(tf.math.argmax(true_w_cos_similarity, axis=1), num_true)
    if true_logits_onehot.dtype != true_logits.dtype:
      true_logits_onehot = tf.dtypes.cast(true_logits_onehot, true_logits.dtype)

    # we want to make the true_w vectors away from each other
    true_w_dist_loss = _get_true_w_dist_loss(tf.reshape(true_w, new_true_w_shape), true_logits_onehot)


    # remove unselectd labels
    labels = tf.dtypes.cast(tf.reduce_sum(labels * tf.dtypes.cast(true_logits_onehot, labels.dtype), 1, keepdims=True),
                            labels.dtype)
    true_logits = tf.dtypes.cast(tf.reduce_sum(true_logits * true_logits_onehot, 1, keepdims=True),
                            true_logits.dtype)
    true_logits_onehot = tf.ones_like(true_logits)

    # only remove accidental hits after remove unselected items in labels
    if remove_accidental_hits:
      sampled_logits = _remove_accidental_hits_from_sampled(labels,
                                         sampled,
                                         sampled_logits,
                                         num_sampled,
                                         1,
                                         )

    # compute logits for the possible translations in foreign language 
    # if labels_dict is given.
    if labels_dict is not None:
      labels_dict_logits, labels_dict_logits_onehot, sampled_logits = \
          _get_labels_dict_logits(labels_dict,
                                  weights,
                                  biases,
                                  partition_strategy,
                                  inputs,
                                  remove_accidental_hits,
                                  sampled,
                                  sampled_logits,
                                  num_sampled,
                                  context_enabled=context_enabled,
                                  valid_sample_index=valid_sample_index,
                                  cluster_proj=cluster_proj,
                                  softmax_sense_all=softmax_sense_all,
                                  tk_context=tk_context)
      out_logits = tf.concat([true_logits, labels_dict_logits, sampled_logits], 1)
      onehot_labels = tf.concat([
          true_logits_onehot,
          labels_dict_logits_onehot,
      ], 1)
      sum_row = tf.reduce_sum(onehot_labels, 1, keepdims=True)
      onehot_labels = onehot_labels/sum_row
      out_labels = tf.concat([
          onehot_labels,
          tf.zeros_like(sampled_logits)
      ], 1)
    else:
      # Construct output logits and labels. The true labels/logits start at col 0.
      out_logits = tf.concat([true_logits, sampled_logits], 1)
      out_labels = tf.concat([
          true_logits_onehot,
          tf.zeros_like(sampled_logits)
      ], 1)

    return out_logits, out_labels, labels, true_w_dist_loss
