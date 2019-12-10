import tensorflow as tf
from checkbox_support.lib import input
from tensorflow.python.ops import rnn
from tensorflow.contrib.seq2seq import hardmax
from bert import transformer_model
import bert
import numpy as np
import os, math
from custom_AttentionWrapper import PointerWrapper

class RelationExtraction(object):
    def __init__(self, args):
        self.is_train = args["is_train"]
        self.batch_size = args["batch_size"]
        self.keep_pob = args["keep_prob"]
        self.dropout_prob = 1.0 - self.keep_pob
        self.learning_rate = args["learning_rate"]

        self.relation_vocab_size = args["relation_vocab_size"]
        self.entity_vocab_size = args["entity_vocab_size"]
        self.entity_type_emb_size = args["entity_emb_size"]
        self.char_vocab_size = args["char_vocab_size"]
        self.char_emb_size = args["char_emb_size"]
        self.sentence_emb_size= args["sentence_emb_size"]
        self.position_emb_size = args["position_emb_size"]

        self.charemb_rnn_hidden = args["charemb_rnn_hidden"]
        self.tokenemb_rnn_hidden = args["tokenemb_rnn_hidden"]

        self.max_sentences = args["max_sentences"]
        self.word_maxlen = args["word_maxlen"]
        self.word_emb_table = args["embedding_table"]
        self.word_emb_size = args["word_emb_size"]

        self.filter_size = args["filter_size"]
        self.num_filter = args["num_filter"]

        self.max_entities = args["max_entities"]
        self.entity_max_tokens = args["entity_max_tokens"]
        self.entity_max_chars = args["entity_max_chars"]
        self.max_relations = args["max_relations"]
        self.max_relation_entities = args["max_relation_entities"]

        # 인코더, 디코더 파라미터
        self.encoder_stack = args["encoder_stack"]
        self.encoder_max_step = args["encoder_max_step"]
        self.encoder_hidden = args["encoder_hidden"]
        self.decoder_hidden = args["decoder_hidden"]
        self.decoder_maxlen = self.max_relations * 3
        self.attention_hidden = args["attention_hidden"]

        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)

        self._placeholder_init()

        finetune_table = tf.get_variable(name="word_embedding_table_finetuning", initializer=self.word_emb_table,
                                         trainable=True, dtype=tf.float32)
        fix_table = tf.get_variable(name="word_embedding_table_fix", initializer=self.word_emb_table,
                                    trainable=False, dtype=tf.float32)
        char_emb_table = tf.get_variable("char_emb_table", shape=[self.char_vocab_size, self.char_emb_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        entity_type_emb_table = tf.get_variable("entity_type_emb_table",
                                                shape=[self.entity_vocab_size, self.entity_type_emb_size],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1))

        sentence_id_emb_table = tf.eye(num_rows=self.max_sentences)

        context_embedding = self._context_embedding_layer(fix_table=fix_table, finetune_table=finetune_table,
                                                          char_emb_table=char_emb_table)
        entity_type_embedding = tf.nn.embedding_lookup(entity_type_emb_table, self.context_entity_type)
        sentence_id_embedding = tf.nn.embedding_lookup(sentence_id_emb_table, self.sentence_id)

        # entity token, character, type, position, sentence_id embedding
        entity_embedding = self._entity_pool_embedding(fix_table=fix_table, finetune_table=finetune_table,
                                                       char_emb_table=char_emb_table,
                                                       token_entities=self.entity_pool,
                                                       char_entities=self.char_entity_pool)

        context_entity_emb = []
        unstack_entity_pool = tf.unstack(entity_embedding, axis=0)
        unstack_context_entity_id = tf.unstack(self.context_entity_id, axis=0)
        for entity_pool, context in zip(unstack_entity_pool, unstack_context_entity_id):
            context_entity_emb.append(tf.nn.embedding_lookup(entity_pool, context))

        context_entity_emb = tf.stack(context_entity_emb, axis=0)
        # context token, character, entity_type, sentence_id embedding
        context_embedding = tf.concat([context_embedding, entity_type_embedding, sentence_id_embedding, context_entity_emb],
                                      axis=-1)

        entity_pool_type_emb = tf.nn.embedding_lookup(entity_type_emb_table, self.entity_pool_type)
        entity_pool_sent_emb = tf.nn.embedding_lookup(sentence_id_emb_table, self.entity_sent_id)

        entity_pool_emb = tf.concat([entity_embedding, entity_pool_type_emb, entity_pool_sent_emb], axis=-1)

        none_emb = tf.get_variable(name="none_emb", shape=[self.decoder_hidden]
                                   , initializer=tf.zeros_initializer)
        pad_emb = tf.get_variable(name="pad_emb", shape=[self.decoder_hidden]
                                  , initializer=tf.zeros_initializer)

        pad_token = tf.expand_dims(tf.stack([pad_emb] * self.batch_size, 0), axis=1, name="pad_token")
        none_token = tf.expand_dims(tf.stack([none_emb] * self.batch_size, 0), axis=1, name="none_token")

        encoder_output, encoder_state = self._biGRU_encoding_layer(encoder_input=context_embedding,
                                                                   encoder_length=self.context_input_length,
                                                                   name="encoder_layer")

        pointing_mem, decoder_state = self._entity_encoding_layer(entity_pool_emb, encoder_output, encoder_state)

        self.pointing_target = tf.concat([pad_token, none_token, pointing_mem], axis=1)
        decoder_input = tf.concat([entity_pool_emb, pointing_mem], axis=-1)

        self._decoder_layer_v3(decoder_input=decoder_input, decoder_init_state=decoder_state,
                               decoder_hidden=self.decoder_hidden, pointing_memory=self.pointing_target)

    def _placeholder_init(self):
        with tf.name_scope("define_placeholder"):
            self.context_input = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.encoder_max_step],
                                                name="context_input")
            self.context_input_length = tf.placeholder(dtype=tf.int32, shape=[self.batch_size],
                                                       name="context_input_length")
            self.context_entity_type = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.encoder_max_step],
                                                      name="context_entity_type")
            self.context_entity_id = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.encoder_max_step],
                                                    name="context_entity_id")
            self.sentence_id = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.encoder_max_step],
                                              name="sentence_id")

            self.char_context_input = tf.placeholder(dtype=tf.int32,
                                                     shape=[self.batch_size, self.encoder_max_step, self.word_maxlen],
                                                     name="char_context_input")
            self.char_context_input_length = tf.placeholder(dtype=tf.int32,
                                                            shape=[self.batch_size, self.encoder_max_step],
                                                            name="char_context_input_length")

            self.entity_pool = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_entities, self.entity_max_tokens],
                                              name="entity_pool")
            self.entity_pool_length = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_entities],
                                                     name="entity_pool_length")
            self.entity_pool_type = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_entities],
                                                   name="entity_pool_length")
            self.entity_sent_id = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_entities],
                                                 name="entity_sent_id")

            self.entity_seq_len = tf.reduce_sum(tf.to_int32(tf.not_equal(self.entity_pool_type,
                                                                         tf.zeros([self.batch_size, self.max_entities],
                                                                                  dtype=tf.int32))) ,axis=-1)
            self.entity_seq_mask = tf.to_float(tf.not_equal(self.entity_pool_type,
                                                            tf.zeros([self.batch_size, self.max_entities],
                                                                     dtype=tf.int32)))

            self.char_entity_pool = tf.placeholder(dtype=tf.int32,
                                                   shape=[self.batch_size, self.max_entities, self.entity_max_chars],
                                                   name="char_entity_pool")
            self.char_entity_pool_length = tf.placeholder(dtype=tf.int32,
                                                          shape=[self.batch_size, self.max_entities],
                                                          name="char_entity_pool_length")

            # self.sb_target = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_relations],
            #                                 name="sb_target")
            # self.re_target = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_relations],
            #                                 name="re_target")
            # self.ob_target = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_relations],
            #                                 name="ob_target")
            # self.entity_target = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_relation_entities],
            #                                 name="entity_target")
            #
            # self.sb_weight = tf.to_float(tf.not_equal(self.sb_target,
            #                                           tf.zeros([self.batch_size, self.max_relations], dtype=tf.int32)))
            # self.re_weight = tf.to_float(tf.not_equal(self.re_target,
            #                                           tf.zeros([self.batch_size, self.max_relations], dtype=tf.int32)))
            # self.entity_weight = tf.to_float(tf.not_equal(self.entity_target,
            #                                               tf.zeros([self.batch_size, self.max_relation_entities],
            #                                                        dtype=tf.int32)))

            self.object_target = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_entities],
                                                name="decoder3_entity_target")
            self.subject_target = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_entities],
                                                 name="decoder3_entity_target")
            self.relation_target = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_entities],
                                                  name="decoder3_relation_target")
            self.rev_relation_target = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_entities],
                                                      name="decoder3_relation_target")

            self.relation_weight = tf.to_float(tf.not_equal(self.object_target,
                                                            tf.ones([self.batch_size, self.max_entities],
                                                                    dtype=tf.int32)))
            self.relation_value_weight = tf.to_float(tf.not_equal(self.object_target,
                                                                  tf.zeros([self.batch_size, self.max_entities],
                                                                           dtype=tf.int32)))

            relation_ones_weight = tf.to_float(tf.equal(self.object_target,
                                                        tf.ones([self.batch_size, self.max_entities],
                                                                dtype=tf.int32))) * 0.02
            self.relation_weight *= self.relation_value_weight
            self.relation_weight += relation_ones_weight

            self.rev_relation_weight = tf.to_float(tf.not_equal(self.subject_target,
                                                                tf.ones([self.batch_size, self.max_entities],
                                                                        dtype=tf.int32)))
            self.rev_relation_value_weight = tf.to_float(tf.not_equal(self.subject_target,
                                                                      tf.zeros([self.batch_size, self.max_entities],
                                                                               dtype=tf.int32)))

            rev_relation_ones_weight = tf.to_float(tf.equal(self.subject_target,
                                                            tf.ones([self.batch_size, self.max_entities],
                                                                    dtype=tf.int32))) * 0.02
            self.rev_relation_weight *= self.rev_relation_value_weight
            self.rev_relation_weight += rev_relation_ones_weight

    def _make_relation_matrix(self, num_relation, emb_size):
        with tf.variable_scope(name_or_scope="relation_matrix"):
            relation_matrix = tf.get_variable(name="relation_emb", shape=[emb_size, emb_size, num_relation],
                                              initializer=tf.truncated_normal_initializer)
            return relation_matrix

    def _context_embedding_layer(self, fix_table, finetune_table, char_emb_table):
        with tf.variable_scope(name_or_scope="context_embedding_layer"):
            _word_emb_fix = tf.nn.embedding_lookup(fix_table, self.context_input)
            _word_emb_finetune = tf.nn.embedding_lookup(finetune_table, self.context_input)

            char_emb = tf.nn.embedding_lookup(char_emb_table, self.char_context_input)
            char_token_emb = self._cnn_char_emb(char_emb, name="char_emb")
            context_emb = tf.concat([_word_emb_fix, _word_emb_finetune, char_token_emb], -1)

            return context_emb

    def _entity_pool_embedding(self, fix_table, finetune_table, char_emb_table, token_entities, char_entities):
        with tf.variable_scope("entity_embedding_layer"):
            fix_emb = tf.nn.embedding_lookup(fix_table, token_entities)
            finetune_emb = tf.nn.embedding_lookup(finetune_table, token_entities)

            token_emb = tf.concat([fix_emb, finetune_emb], -1)
            entity_token_emb = self._token_cnn_emb(token_emb, name="entity_token_emb")

            char_emb = tf.nn.embedding_lookup(char_emb_table, char_entities)
            entity_char_emb = self._cnn_char_emb(char_emb, name="char_emb")

            entity_emb = tf.concat([entity_token_emb, entity_char_emb], -1)

            return entity_emb

    def _token_cnn_emb(self, inputs, name):
        with tf.variable_scope(name):
            # 풀링 된 벡터 값들 저장할 리스트
            _pooled_outputs = []

            emb_size = inputs.shape[-1]

            # 필터사이즈 별로 Convolution 계층 설계
            for filter in self.filter_size:
                # 필터 선언
                _convolution_w = tf.get_variable(name=str(filter)+"convolution_w",
                                                 shape=[1, filter, emb_size, self.num_filter],
                                                 initializer=tf.contrib.layers.xavier_initializer())
                _convolution_b = tf.get_variable(name=str(filter)+"convolution_bias", shape=[self.num_filter],
                                                 initializer=tf.contrib.layers.xavier_initializer())

                # Convolution 진행
                _conv = tf.nn.conv2d(inputs, _convolution_w, strides=[1, 1, 1, 1],
                                     padding='VALID', name="convolution_result")

                # 활성함수 적용
                _h = tf.nn.leaky_relu(tf.nn.bias_add(_conv, _convolution_b), name="leaky_relu")

                # max pooling
                _pooled = tf.reduce_max(_h, 2)  # [batch, step, filter_size]
                # pooling 결과값 리스트에 저장
                _pooled_outputs.append(_pooled)

            # 어절 임베딩 concat
            emb_output = tf.concat(_pooled_outputs, axis=2)  # batch x step x convolution_embedding size
        return emb_output

    def _cnn_char_emb(self, embedding, name):
        '''
        어절 단위 임베딩 수행
        어절을 이루는 형태소들을 CNN에 통과시켜 어절단위 임베딩 출력
        :param
        morp_emb: 형태소 임베딩, [ batch, max_step, word_max_length, morp_emb_size(단어 임베딩 벡터 사이즈 * 2 + 태그 임베딩 벡터 사이즈) ]
        :return:
        emb_output: 어절 임베딩
        [ batch, max_step, token_emb_size(filter_size 갯수 * num_filter) ]
        '''
        emb_size = embedding.shape[-1]
        with tf.variable_scope(name):
            # 풀링 된 벡터 값들 저장할 리스트
            _pooled_outputs = []
            # 형태소 임베딩 사이즈 (형태소 임베딩 벡터 사이즈 * 2 + 태그 임베딩 벡터 사이즈)

            # 필터사이즈 별로 Convolution 계층 설계
            for filter in self.filter_size:
                # 필터 선언
                _convolution_w = tf.get_variable(name=str(filter)+"convolution_w",
                                                 shape=[1, filter, emb_size, self.num_filter],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
                _convolution_b = tf.get_variable(name=str(filter)+"convolution_bias", shape=[self.num_filter],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.1))

                # Convolution 진행
                _conv = tf.nn.conv2d(embedding, _convolution_w, strides=[1, 1, 1, 1],
                                     padding='VALID', name="convolution_result")

                # 활성함수 적용
                _h = tf.nn.leaky_relu(tf.nn.bias_add(_conv, _convolution_b), name="relu")

                # max pooling
                _pooled = tf.reduce_max(_h, 2)  # [batch, step, filter_size]
                # pooling 결과값 리스트에 저장
                _pooled_outputs.append(_pooled)

            # 어절 임베딩 concat
            emb_output = tf.concat(_pooled_outputs, axis=2)  # batch x step x convolution_embedding size
        return emb_output

    def _biGRU_encoding_layer(self, encoder_input, encoder_length, name, num_stack=1):
        '''
        RNN으로 인코딩하는 계층, 양방향 GRU 사용
        :param
        encoder_input: 어절 단위 임베딩 [ batch, max_step, token_emb_size ]
        :return:
        encoder_output_concat: 인코더 출력 [ batch, max_step, (encoder_hidden_size * 2) ]
        encoder_state_concat: 인코더 state [ batch, (encoder_hidden_size * 2) ]
        '''
        with tf.variable_scope(name_or_scope=name):
            # 멀티 레이어 사용 유무에 따라서 GRU cell 선언
            if num_stack == 1:
                print("Encoder mode: single RNN")
                _fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.encoder_hidden,
                                                                 dropout_keep_prob=self.keep_pob)
                _bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.encoder_hidden,
                                                                 dropout_keep_prob=self.keep_pob)
            else:
                print("Encoder mode: Stacked RNN")
                print("num stack: ", self.encoder_stack)
                _multi_fw_cells = [tf.contrib.rnn.LayerNormBasicLSTMCell(self.encoder_hidden,
                                                                         dropout_keep_prob=self.keep_pob) for _ in range(num_stack)]
                _multi_bw_cells = [tf.contrib.rnn.LayerNormBasicLSTMCell(self.encoder_hidden,
                                                                         dropout_keep_prob=self.keep_pob) for _ in range(num_stack)]

                _fw_cell = tf.contrib.rnn.MultiRNNCell(_multi_fw_cells, state_is_tuple=False)
                _bw_cell = tf.contrib.rnn.MultiRNNCell(_multi_bw_cells, state_is_tuple=False)

            encoder_output, encoder_state = rnn.bidirectional_dynamic_rnn(
                cell_fw=_fw_cell,
                cell_bw=_bw_cell,
                inputs=encoder_input,
                sequence_length=encoder_length,
                dtype=tf.float32, time_major=False
            )
            # output, state: 양방향이라서 두개 나옴. concat 한다.
            # output: [ batch * max_step * (encoder_hidden_size * 2) ]
            encoder_output_concat = tf.concat(encoder_output, -1)
            # state: [ batch * (encoder_hidden_size * 2) ]
            encoder_state_concat = tf.concat(encoder_state, -1)

        return encoder_output_concat, encoder_state

    def _multi_head_attention(self, key, query, value, attention_name, num_heads=8, head_size=32, intermediate_size=512,
                              return_type="concat"):
        with tf.variable_scope(name_or_scope=attention_name):
            _query = tf.layers.dense(query, units=num_heads*head_size, activation=tf.nn.leaky_relu, name="query")
            _key = tf.layers.dense(key, units=num_heads*head_size, activation=tf.nn.leaky_relu, name="key")
            _value = tf.layers.dense(value, units=num_heads*head_size, activation=tf.nn.leaky_relu, name="value")

            _query_split = tf.split(_query, num_heads, axis=-1)
            _key_split = tf.split(_key, num_heads, axis=-1)
            _value_split = tf.split(_value, num_heads, axis=-1)

            _query_split = [tf.layers.dense(q, head_size, activation=tf.nn.leaky_relu) for q in _query_split]
            _key_split = [tf.layers.dense(k, head_size, activation=tf.nn.leaky_relu) for k in _key_split]
            _value_split = [tf.layers.dense(v, head_size, activation=tf.nn.leaky_relu) for v in _value_split]

            _query_concat = tf.concat(_query_split, axis=0)
            _key_concat = tf.concat(_key_split, axis=0)
            _value_concat = tf.concat(_value_split, axis=0)

            _matmul_query_key = tf.matmul(_query_concat, _key_concat, transpose_b=True)
            _scale_align = _matmul_query_key / (head_size ** 0.5)
            _softmax_align = tf.nn.softmax(_scale_align, -1)

            _output = tf.matmul(_softmax_align, _value_concat)

            # query_step * key_step
            multi_head_align = tf.add_n(tf.split(_scale_align, num_heads, axis=0))
            multi_head_output = tf.concat(tf.split(_output, num_heads, axis=0), axis=2)

            # multi_head_output = tf.layers.dense(multi_head_output, intermediate_size, activation=tf.nn.leaky_relu,
            #                                     name="mh_out")
            # query = tf.layers.dense(query, intermediate_size, activation=tf.nn.leaky_relu, name="query_intermediate")

            if return_type == "concat":
                residual_output = tf.concat([multi_head_output, query], axis=-1)
            elif return_type == "dense":
                residual_output = tf.concat([multi_head_output, query], axis=-1)
                residual_output = tf.layers.dense(residual_output, intermediate_size, activation=tf.nn.leaky_relu,
                                                  name="mh_output")
            elif return_type == "residual":
                residual_output = multi_head_output + query
            else:
                residual_output = multi_head_output

        return residual_output, multi_head_align

    def _context2relation_attention(self, context, relations, return_type, name):
        with tf.variable_scope(name_or_scope=name):
            c2r_att_matrix = tf.matmul(context, relations, transpose_b=True)
            c2r_att_softmax = tf.nn.softmax(c2r_att_matrix)

            att_output = tf.matmul(c2r_att_softmax, relations)

            if return_type == "concat":
                return tf.concat([context, att_output], -1), c2r_att_matrix
            else:
                return context+att_output, c2r_att_matrix

    def biaffine_attention(self, subject, object, relation_matrix):
        with tf.variable_scope("biaffine_attention", reuse=tf.AUTO_REUSE):
            # sub, ob : [b, h]
            # relation matrix: [h, h, r]
            batch, hidden, relation = self.batch_size, self.decoder_hidden, self.relation_vocab_size

            # [h, h, r] --> [h, h*r]
            reshape_matrix = tf.reshape(relation_matrix, shape=[hidden, hidden*relation])
            # [b, h] * [h, h*r] --> [b, h*r]
            sub_rel_att = tf.matmul(subject, reshape_matrix)
            # [b, h*r] --> [b, r, h]
            reshape_att = tf.transpose(tf.reshape(sub_rel_att, shape=[batch, hidden, relation]), perm=[0, 2, 1])
            # [b, r, h] * [b, h, 1] --> [b, r, 1]
            expand_ob = tf.expand_dims(object, axis=1)
            bi_attention = tf.matmul(reshape_att, expand_ob, transpose_b=True)

            # [b, r, 1] --> [b, r]
            output = tf.squeeze(bi_attention, axis=-1)
            bias = tf.get_variable(name="biaffine_att_bias", shape=[relation], initializer=tf.zeros_initializer)

            return output + bias

    def _bahdanau_attenion(self, key, query, attention_name):
        with tf.variable_scope(attention_name):
            dtype = query.dtype
            units = query.get_shape()[-1]

            expand_query = tf.expand_dims(query, axis=1)
            num_units = key.shape[2].value or tf.shape(key)[2]
            v = tf.get_variable("attention_v", [num_units], dtype=dtype)
            g = tf.get_variable("attention_g", dtype=dtype,
                                initializer=tf.constant_initializer(math.sqrt((1. / num_units))), shape=())
            b = tf.get_variable(
                "attention_b", [num_units], dtype=dtype,
                initializer=tf.zeros_initializer())
            normed_v = g * v * tf.rsqrt(tf.reduce_sum(tf.square(v)))

            return tf.reduce_sum(normed_v * tf.tanh(key + expand_query + b), [2])

    def _entity_encoding_layer(self, entity_input, context_encoder_output, encoder_state):
        with tf.variable_scope("entity_encoding_layer"):
            sequence_len = [self.max_entities] * self.batch_size
            memory = context_encoder_output

            encoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.decoder_hidden,
                                                                 dropout_keep_prob=self.keep_pob)

            # mechanism = tf.contrib.seq2seq.LuongAttention(self.decoder_hidden, memory,
            #                                               memory_sequence_length=self.context_input_length)
            # encoder_cell = tf.contrib.seq2seq.AttentionWrapper(encoder_cell, mechanism,
            #                                                    attention_layer_size=self.decoder_hidden,
            #                                                    alignment_history=False)

            encoder_init_state = encoder_cell.zero_state(self.batch_size, dtype=tf.float32)

            helper = tf.contrib.seq2seq.TrainingHelper(entity_input, sequence_len)
            encoder = tf.contrib.seq2seq.BasicDecoder(cell=encoder_cell, helper=helper,
                                                      initial_state=encoder_init_state)

            encoder_output, state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=encoder,
                                                                         maximum_iterations=self.max_entities)

            output, _ = self._multi_head_attention(key=memory, query=encoder_output.rnn_output, value=memory, head_size=64,
                                                   attention_name="context2entity_attention")
            output = tf.layers.dense(output, self.decoder_hidden, activation=tf.nn.leaky_relu)

            return output, state

    def _decoder_layer_v3(self, decoder_input, decoder_init_state, decoder_hidden, pointing_memory):
        with tf.variable_scope("decoder_v3"):
            init_state = decoder_init_state

            with tf.variable_scope("object_cell_define"):
                object_decoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(decoder_hidden, dropout_keep_prob=self.keep_pob)
                object_cell_pre_state = init_state
                # decoder_cell = PointerWrapper(decoder_cell, self.attention_hidden,
                #                                      self.entity_pool_embedding,
                #                                      initial_cell_state=init_state)
            with tf.variable_scope("subject_cell_define"):
                subject_decoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(decoder_hidden, dropout_keep_prob=self.keep_pob)
                subject_cell_pre_state = init_state

            with tf.variable_scope("decoder_input_layer"):
                # mh_output, _ = self._multi_head_attention(key=self.self_att_output, query= self.entity_pool_embedding,
                #                                        value=self.self_att_output, attention_name="etity2context_att")
                # mh_output_step = tf.unstack(mh_output, axis=1)
                decoder_input_per_step = tf.unstack(decoder_input, axis=1)

            with tf.variable_scope("decoding_triple", reuse=tf.AUTO_REUSE):
                object_logits = []
                relation_logits = []
                subject_logits = []
                rev_relation_logits = []

                for i in range(self.max_entities):
                    input = decoder_input_per_step[i]
                    object_deocder_output, object_state = object_decoder_cell(input, object_cell_pre_state)
                    subject_decoder_output, subject_state = subject_decoder_cell(input, subject_cell_pre_state)

                    # object_pointing = self._bahdanau_attenion(pointing_memory, object_deocder_output,
                    #                                           attention_name="object_pointing")
                    # subject_pointing = self._bahdanau_attenion(pointing_memory, subject_decoder_output,
                    #                                            attention_name="subject_pointing")

                    object_deocder_output = tf.expand_dims(object_deocder_output, axis=1)
                    subject_decoder_output = tf.expand_dims(subject_decoder_output, axis=1)

                    relation_output, object_pointing = self._multi_head_attention(key=pointing_memory,
                                                                                  query=object_deocder_output,
                                                                                  value=pointing_memory,
                                                                                  attention_name="object_pointing")
                    rev_output, subject_pointing = self._multi_head_attention(key=pointing_memory,
                                                                              query=subject_decoder_output,
                                                                              value=pointing_memory,
                                                                              attention_name="subject_pointing")
                    object_pointing = tf.squeeze(object_pointing, axis=1)
                    subject_pointing = tf.squeeze(subject_pointing, axis=1)
                    relation_output = tf.squeeze(relation_output, axis=1)
                    rev_output = tf.squeeze(rev_output, axis=1)

                    relation_logit = tf.layers.dense(relation_output, units=self.relation_vocab_size,
                                                     activation=tf.nn.leaky_relu, name="relation_label")
                    rev_relation_logit = tf.layers.dense(rev_output, units=self.relation_vocab_size,
                                                         activation=tf.nn.leaky_relu, name="rev_relation_label")
                    object_logits.append(object_pointing)
                    relation_logits.append(relation_logit)
                    subject_logits.append(subject_pointing)
                    rev_relation_logits.append(rev_relation_logit)

                    object_cell_pre_state = object_state
                    subject_cell_pre_state = subject_state

                object_logits = tf.stack(object_logits, axis=1)
                relation_logits = tf.stack(relation_logits, axis=1)
                subject_logits = tf.stack(subject_logits, axis=1)
                rev_relation_logits = tf.stack(rev_relation_logits, axis=1)

                self.object_predicts = tf.argmax(object_logits, axis=-1)
                self.relation_predicts = tf.argmax(relation_logits, axis=-1)
                self.subject_predicts = tf.argmax(subject_logits, axis=-1)
                self.rev_relation_predicts = tf.argmax(rev_relation_logits, axis=-1)

            with tf.variable_scope("training_layer"):
                self.object_loss = tf.losses.sparse_softmax_cross_entropy(logits=object_logits,
                                                                          labels=self.object_target,
                                                                          weights=self.relation_weight)
                self.re_loss = tf.losses.sparse_softmax_cross_entropy(logits=relation_logits,
                                                                      labels=self.relation_target,
                                                                      weights=self.relation_weight)
                self.subject_loss = tf.losses.sparse_softmax_cross_entropy(logits=subject_logits,
                                                                           labels=self.subject_target,
                                                                           weights=self.rev_relation_weight)
                self.rev_re_loss = tf.losses.sparse_softmax_cross_entropy(logits=rev_relation_logits,
                                                                          labels=self.rev_relation_target,
                                                                          weights=self.rev_relation_weight)
                # self.object_loss = tf.contrib.seq2seq.sequence_loss(logits=object_logits,
                #                                                       targets=self.decoder3_entity_target,
                #                                                       weights=self.decoder3_weight)
                # self.re_loss = tf.contrib.seq2seq.sequence_loss(logits=relation_logits,
                #                                                       targets=self.decoder3_relation_target,
                #                                                       weights=self.decoder3_weight)
                self.object_loss = tf.reduce_mean(self.object_loss)
                self.re_loss = tf.reduce_mean(self.re_loss)
                self.subject_loss = tf.reduce_mean(self.subject_loss)
                self.rev_re_loss = tf.reduce_mean(self.rev_re_loss)

                self.loss = (0.4*self.object_loss) + (0.4*self.subject_loss) + (0.1*self.re_loss) + (0.1*self.rev_re_loss)

                _optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self._gradients = _optimizer.compute_gradients(self.loss)
                for g in self._gradients:
                    print(g)
                _apply_op = _optimizer.apply_gradients(self._gradients, global_step=self.global_step)
                _ema = tf.train.ExponentialMovingAverage(decay=0.9999)

                with tf.control_dependencies([_apply_op]):
                    _ema_op = _ema.apply(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
                    self.train_op = tf.group(_ema_op)

                self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    def _decoder_layer_v4(self, decoder_input, decoder_init_state, decoder_hidden, pointing_memory):
        with tf.variable_scope("decoder_v4"):
            self.relation_table = self._make_relation_matrix(self.relation_vocab_size, self.decoder_hidden)
            init_state = decoder_init_state

            def get_entity_emb_from_pointer(entity_emb, pointer):
                result = []
                emb_per_batch = tf.unstack(entity_emb, axis=0)
                idx_per_batch = tf.unstack(tf.argmax(pointer, axis=-1), axis=0)

                for emb, pointing in zip(emb_per_batch, idx_per_batch):
                    result.append(tf.nn.embedding_lookup(emb, pointing))

                result = tf.stack(result, axis=0)
                # result = tf.layers.dense(tf.stack(result, axis=0), self.decoder_hidden, activation=tf.nn.leaky_relu)

                return result

            with tf.variable_scope("object_cell_define"):
                object_decoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(decoder_hidden, dropout_keep_prob=self.keep_pob)
                object_cell_pre_state = init_state

            with tf.variable_scope("subject_cell_define"):
                subject_decoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(decoder_hidden, dropout_keep_prob=self.keep_pob)
                subject_cell_pre_state = init_state

            with tf.variable_scope("decoder_input_layer"):
                decoder_input_per_step = tf.unstack(decoder_input, axis=1)

            with tf.variable_scope("decoding_triple", reuse=tf.AUTO_REUSE):
                object_logits = []
                relation_logits = []
                subject_logits = []
                rev_relation_logits = []

                for i in range(self.max_entities):
                    input = decoder_input_per_step[i]
                    object_deocder_output, object_state = object_decoder_cell(input, object_cell_pre_state)
                    subject_decoder_output, subject_state = subject_decoder_cell(input, subject_cell_pre_state)

                    object_deocder_output = tf.expand_dims(object_deocder_output, axis=1)
                    subject_decoder_output = tf.expand_dims(subject_decoder_output, axis=1)

                    _, object_pointing = self._multi_head_attention(key=pointing_memory,
                                                                                  query=object_deocder_output,
                                                                                  value=pointing_memory,
                                                                                  attention_name="object_pointing")
                    _, subject_pointing = self._multi_head_attention(key=pointing_memory,
                                                                              query=subject_decoder_output,
                                                                              value=pointing_memory,
                                                                              attention_name="subject_pointing")
                    object_pointing = tf.squeeze(object_pointing, axis=1)
                    subject_pointing = tf.squeeze(subject_pointing, axis=1)

                    object_emb = get_entity_emb_from_pointer(self.pointing_target, object_pointing)
                    subject_emb = get_entity_emb_from_pointer(self.pointing_target, subject_pointing)

                    biaffine_input = tf.layers.dense(input, self.decoder_hidden, activation=tf.nn.leaky_relu)

                    relation_logit = self.biaffine_attention(biaffine_input, object_emb, self.relation_table)
                    rev_relation_logit = self.biaffine_attention(subject_emb, biaffine_input, self.relation_table)

                    object_logits.append(object_pointing)
                    relation_logits.append(relation_logit)
                    subject_logits.append(subject_pointing)
                    rev_relation_logits.append(rev_relation_logit)

                    object_cell_pre_state = object_state
                    subject_cell_pre_state = subject_state

                object_logits = tf.stack(object_logits, axis=1)
                relation_logits = tf.stack(relation_logits, axis=1)
                subject_logits = tf.stack(subject_logits, axis=1)
                rev_relation_logits = tf.stack(rev_relation_logits, axis=1)

                self.object_predicts = tf.argmax(object_logits, axis=-1)
                self.relation_predicts = tf.argmax(relation_logits, axis=-1)
                self.subject_predicts = tf.argmax(subject_logits, axis=-1)
                self.rev_relation_predicts = tf.argmax(rev_relation_logits, axis=-1)

            with tf.variable_scope("training_layer"):
                self.object_loss = tf.losses.sparse_softmax_cross_entropy(logits=object_logits,
                                                                          labels=self.object_target,
                                                                          weights=self.relation_weight)
                self.re_loss = tf.losses.sparse_softmax_cross_entropy(logits=relation_logits,
                                                                      labels=self.relation_target,
                                                                      weights=self.relation_weight)
                self.subject_loss = tf.losses.sparse_softmax_cross_entropy(logits=subject_logits,
                                                                           labels=self.subject_target,
                                                                           weights=self.rev_relation_weight)
                self.rev_re_loss = tf.losses.sparse_softmax_cross_entropy(logits=rev_relation_logits,
                                                                          labels=self.rev_relation_target,
                                                                          weights=self.rev_relation_weight)

                self.object_loss = tf.reduce_mean(self.object_loss)
                self.re_loss = tf.reduce_mean(self.re_loss)
                self.subject_loss = tf.reduce_mean(self.subject_loss)
                self.rev_re_loss = tf.reduce_mean(self.rev_re_loss)

                self.loss = (0.4*self.object_loss) + (0.4*self.subject_loss) + (0.1*self.re_loss) + (0.1*self.rev_re_loss)

                _optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self._gradients = _optimizer.compute_gradients(self.loss)
                for g in self._gradients:
                    print(g)
                _apply_op = _optimizer.apply_gradients(self._gradients, global_step=self.global_step)
                _ema = tf.train.ExponentialMovingAverage(decay=0.9999)

                with tf.control_dependencies([_apply_op]):
                    _ema_op = _ema.apply(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
                    self.train_op = tf.group(_ema_op)

                self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    # def test_decoder(self, decoder_init_state, decoder_hidden, decoder_maxlen, start_token):
    #     with tf.name_scope("decoder_layer"):
    #         previous_state = decoder_init_state
    #
    #         sb_logits, re_logits, ob_logits = [], [], []
    #         sb_predict, re_predict, ob_predict = [], [], []
    #         self.sb_input = []
    #
    #         self.entity_pool_embedding = tf.layers.dense(self.entity_pool_embedding, units=self.entity_emb_size,
    #                                                      activation=tf.nn.leaky_relu)
    #
    #         unstack_pool = tf.unstack(self.entity_pool_embedding, axis=0)
    #         unstack_sb = tf.unstack(self.sb_target, axis=0)
    #         unstack_ob = tf.unstack(self.ob_target, axis=0)
    #
    #         sb_emb_pool = []
    #         ob_emb_pool = []
    #         for idx, _ in enumerate(unstack_pool):
    #             sb_emb_pool.append(tf.nn.embedding_lookup(unstack_pool[idx], unstack_sb[idx]))
    #             ob_emb_pool.append(tf.nn.embedding_lookup(unstack_pool[idx], unstack_ob[idx]))
    #
    #         sb_emb = tf.stack(sb_emb_pool, axis=0)
    #         ob_emb = tf.stack(ob_emb_pool, axis=0)
    #         re_emb = tf.nn.embedding_lookup(self.relation_matrix, self.re_target)
    #
    #         sb_emb_by_step = tf.unstack(sb_emb, axis=1)
    #         ob_emb_by_step = tf.unstack(ob_emb, axis=1)
    #         re_emb_by_step = tf.unstack(re_emb, axis=1)
    #
    #         object_decoder_cell = tf.contrib.rnn.GRUCell(decoder_hidden)
    #         object_decoder_cell = tf.contrib.rnn.DropoutWrapper(object_decoder_cell, input_keep_prob=self.keep_pob,
    #                                                             output_keep_prob=self.keep_pob)
    #
    #         all_state = [previous_state]
    #         next_input = start_token
    #
    #         with tf.variable_scope("subject_test", reuse=tf.AUTO_REUSE):
    #             for i in range(self.max_relations):
    #                 if i != 0:
    #                     next_input = tf.concat([ob_emb_by_step[i-1], re_emb_by_step[i-1]], axis=-1)
    #
    #                 expand_input = tf.expand_dims(next_input, axis=1)
    #                 object_context, _ = self._multi_head_attention(key=self.self_att_output, query=expand_input,
    #                                                                value=self.self_att_output,
    #                                                                attention_name="object_attention")
    #                 object_context = tf.squeeze(object_context, axis=1)
    #
    #                 state = previous_state
    #                 output, state = object_decoder_cell(object_context, state)
    #
    #                 all_state.append(state)
    #
    #                 pointing_score_object = self._bahdanau_attenion(key=self.entity_pool_embedding, query=output,
    #                                                                  attention_name="pointing_object")
    #
    #                 self.object_logit = pointing_score_object
    #                 object_predict = tf.argmax(pointing_score_object, axis=-1)
    #
    #                 unstack_predict = tf.unstack(object_predict, axis=0)
    #                 ob_pool = []
    #
    #                 for entities, idx in zip(unstack_pool, unstack_predict):
    #                     ob_pool.append(tf.nn.embedding_lookup(entities, idx))
    #
    #                 ob_logits.append(self.object_logit)
    #                 ob_predict.append(object_predict)
    #
    #         # 지배소 예측값
    #         decoding_score = tf.stack(ob_logits, axis=1)
    #
    #         self.states = tf.stack(all_state, axis=1)
    #         self.outputs = decoding_score
    #         self.predict = tf.stack(ob_predict, axis=1)
    #
    #         self.predict_pointer = tf.argmax(decoding_score, axis=-1)
    #
    #         self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decoding_score, labels=self.ob_target)
    #         self.loss = tf.reduce_mean(self.loss)
    #         _optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    #         self._gradients = _optimizer.compute_gradients(self.loss)
    #         for g in self._gradients:
    #             print(g)
    #         _apply_op = _optimizer.apply_gradients(self._gradients, global_step=self.global_step)
    #         _ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    #
    #         with tf.control_dependencies([_apply_op]):
    #             _ema_op = _ema.apply(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    #             self.train_op = tf.group(_ema_op)
    #
    #         self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    # def train_layer(self, logits):
    #     with tf.name_scope("train_layer"):
    #         sb_logits, re_logits, ob_logits = logits
    #         self.sb_loss = tf.contrib.seq2seq.sequence_loss(logits=sb_logits, targets=self.sb_target,
    #                                                         weights=self.sb_weight)
    #         self.re_loss = tf.contrib.seq2seq.sequence_loss(logits=re_logits, targets=self.re_target,
    #                                                         weights=self.re_weight)
    #         self.ob_loss = tf.contrib.seq2seq.sequence_loss(logits=ob_logits, targets=self.ob_target,
    #                                                         weights=self.sb_weight)
    #
    #         self.loss = (0.6*self.sb_loss) + (0.1*self.ob_loss) + (0.3*self.re_loss)
    #         # self.loss = self.sb_loss
    #
    #         _optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    #         self._gradients = _optimizer.compute_gradients(self.loss)
    #         for g in self._gradients:
    #             print(g)
    #         _apply_op = _optimizer.apply_gradients(self._gradients, global_step=self.global_step)
    #         _ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    #
    #         with tf.control_dependencies([_apply_op]):
    #             _ema_op = _ema.apply(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    #             self.train_op = tf.group(_ema_op)
    #
    #         self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    def train_step(self, sess, inputs):
        output = [self.object_loss, self.re_loss, self.subject_loss, self.rev_re_loss, self.loss, self.train_op,
                  self.object_target, self.relation_target, self.object_predicts, self.relation_predicts,
                  self.relation_weight, self.subject_target, self.rev_relation_target, self.rev_relation_weight]

        placeholders = [self.context_input, self.context_input_length, self.context_entity_type, self.context_entity_id,
                        self.sentence_id, self.char_context_input, self.char_context_input_length, self.entity_pool,
                        self.entity_pool_length, self.entity_pool_type, self.entity_sent_id, self.char_entity_pool,
                        self.char_entity_pool_length, self.object_target, self.relation_target, self.subject_target,
                        self.rev_relation_target]

        input_feed = {holder: input for holder, input in zip(placeholders, inputs)}

        # sb_loss, re_loss, ob_loss, total_loss, _, st, sp, rt, rp, ot, op, ee, sin, slo = sess.run(output, feed_dict=input_feed)
        object_loss, re_loss, subject_loss, rev_loss, loss, _, ot, rt, op, rp, w, st, rrt, rw = sess.run(output,
                                                                                                         feed_dict=input_feed)
        np.set_printoptions(threshold=np.nan, linewidth=350)
        # print()
        # print("ot:", ot[0])
        # # print("op:", op[0])
        # print("rt:", rt[0])
        # # print("rp:", rp[0])
        # print(w[0])
        # print("st:", st[0])
        # print("rrt:", rrt[0])
        # print(rw[0])
        # print(len)
        # print("entity emb:\n", ee[5])
        # print("sub target:\n", st[5])
        # print("sub pointer in:\n", sin[5])
        # print("sub pointer out:\n", slo[5])
        #
        # for ts, ps, tr, pr, to, po in zip(st, sp, rt, rp, ot, op):
        #     print("subject_target:")
        #     print(ts)
        #     print("subject_predict:")
        #     print(ps)
        #     print("relation_target:")
        #     print(tr)
        #     print("relation_predict:")
        #     print(pr)
        #     print("object_target:")
        #     print(to)
        #     print("object_predit:")
        #     print(po)
        #     break

        # print("target: \n", target)
        # print("predict: \n", predict)

        # return sb_loss, re_loss, ob_loss, total_loss
        return object_loss, re_loss, subject_loss, rev_loss, loss

    def valid_step(self, sess, inputs):
        # output = [self.sb_loss, self.re_loss, self.ob_loss, self.loss, self.train_op, self.sb_target, self.subject_predicts,
        #           self.re_target, self.relation_predicts, self.ob_target, self.object_predicts,self.entity_pool_embedding,
        #           self.sb_input, self.sb_logit]
        output = [self.object_predicts, self.relation_predicts, self.subject_predicts, self.rev_relation_predicts,
                  self.object_target, self.relation_target]
        # output = [self.sb_target, self.predict, self.outputs, self.loss, self.train_op]

        # placeholders = [self.context_input, self.context_input_length, self.context_entity_type, self.context_entity_id,
        #                 self.sentence_id, self.char_context_input, self.char_context_input_length, self.entity_pool,
        #                 self.entity_pool_length, self.entity_pool_type, self.entity_sent_id, self.char_entity_pool,
        #                 self.char_entity_pool_length, self.sb_target, self.re_target, self.ob_target]

        placeholders = [self.context_input, self.context_input_length, self.context_entity_type, self.context_entity_id,
                        self.sentence_id, self.char_context_input, self.char_context_input_length, self.entity_pool,
                        self.entity_pool_length, self.entity_pool_type, self.entity_sent_id, self.char_entity_pool,
                        self.char_entity_pool_length, self.object_target, self.relation_target]

        input_feed = {holder: input for holder, input in zip(placeholders, inputs)}

        # sb_loss, re_loss, ob_loss, total_loss, _, st, sp, rt, rp, ot, op, ee, sin, slo = sess.run(output, feed_dict=input_feed)
        object_predict, relation_predict, subject_predict, rev_re_predict, et, rt = sess.run(output, feed_dict=input_feed)
        np.set_printoptions(threshold=np.nan, linewidth=200)
        # print(len)
        # print("entity emb:\n", ee[5])
        # print("sub target:\n", st[5])
        # print("sub pointer in:\n", sin[5])
        # print("sub pointer out:\n", slo[5])
        #
        # for ts, ps, tr, pr, to, po in zip(st, sp, rt, rp, ot, op):
        #     print("subject_target:")
        #     print(ts)
        #     print("subject_predict:")
        #     print(ps)
        #     print("relation_target:")
        #     print(tr)
        #     print("relation_predict:")
        #     print(pr)
        #     print("object_target:")
        #     print(to)
        #     print("object_predit:")
        #     print(po)
        #     break

        # print("target: \n", target)
        # print("predict: \n", predict)

        # return sb_loss, re_loss, ob_loss, total_loss
        # print()
        # print("ot:", et[0])
        # print("op:", object_predict[0])
        # print("rt:", rt[0])
        # print("rp:", relation_predict[0])
        return object_predict, relation_predict, subject_predict, rev_re_predict

    def predict_step(self, sess, inputs):
        # output = [self.subject_predicts, self.relation_predicts, self.object_predicts]
        # placeholders = [self.context_input, self.context_input_length, self.context_entity_type, self.context_entity_id,
        #                 self.sentence_id, self.char_context_input, self.char_context_input_length, self.entity_pool,
        #                 self.entity_pool_length, self.entity_pool_type, self.entity_sent_id, self.char_entity_pool,
        #                 self.char_entity_pool_length]
        #
        # input_feed = {holder: input for holder, input in zip(placeholders, inputs)}
        #
        # sb_p, re_p, ob_p = sess.run(output, feed_dict=input_feed)
        #
        # return sb_p, re_p, ob_p
        output = [self.object_predicts, self.relation_predicts, self.subject_predicts, self.rev_relation_predicts]
        placeholders = [self.context_input, self.context_input_length, self.context_entity_type, self.context_entity_id,
                        self.sentence_id, self.char_context_input, self.char_context_input_length, self.entity_pool,
                        self.entity_pool_length, self.entity_pool_type, self.entity_sent_id, self.char_entity_pool,
                        self.char_entity_pool_length]
        input_feed = {holder: input for holder, input in zip(placeholders, inputs)}
        object_predict, relation_predict, subject_predict, rev_relation_predict = sess.run(output, feed_dict=input_feed)

        return object_predict, relation_predict, subject_predict, rev_relation_predict


    def save_model(self, sess, path, epo_num):
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, str(epo_num) + "_model.ckpt")
        print("Saving model, %s" % checkpoint_path)
        self.saver.save(sess, checkpoint_path)

if __name__=="__main__":
    emb_table = np.random.rand(50, 100).astype(dtype=np.float32)

    args = {
        "is_train": False,
        "pad_id":0,
        "start_id":1,
        "end_id": 1,
        "batch_size": 32,
        "keep_prob": 0.8,
        "learning_rate": 0.01,
        "relation_vocab_size": 15,
        "entity_vocab_size": 7,
        "entity_emb_size": 50,
        "char_vocab_size": 200,
        "char_emb_size": 50,
        "charemb_rnn_hidden": 64,
        "tokenemb_rnn_hidden": 128,
        "word_maxlen": 20,
        "embedding_table":emb_table,
        "word_emb_size": 100,
        "max_entities": 10,
        "entity_max_tokens": 5,
        "entity_max_chars": 10,
        "encoder_stack": 1,
        "encoder_max_step": 50,
        "encoder_hidden": 64,
        "decoder_hidden": 128,
        "max_relations": 10,
        "attention_hidden": 128,
    }

    model = RelationExtraction(args)