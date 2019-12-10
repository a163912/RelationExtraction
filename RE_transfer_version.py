import tensorflow as tf
from tensorflow.python.ops import rnn
import numpy as np
import os, math

class RelationExtraction(object):
    def __init__(self, args):
        '''
        모델 초기화
        :param args: 하이퍼 파라미터가 저장된 dict
        '''
        self.is_train = args["is_train"]
        self.batch_size = args["batch_size"]
        self.keep_pob = args["keep_prob"]
        self.dropout_prob = 1.0 - self.keep_pob
        self.learning_rate = args["learning_rate"]

        self.relation_vocab_size = args["relation_vocab_size"]
        self.entity_vocab_size = args["entity_vocab_size"]
        self.entity_type_emb_size = args["entity_type_emb_size"]
        self.char_vocab_size = args["char_vocab_size"]
        self.char_emb_size = args["char_emb_size"]

        self.max_sentences = args["max_sentences"]
        self.word_maxlen = args["word_maxlen"]
        self.word_emb_table = args["embedding_table"]
        self.word_emb_size = args["word_emb_size"]

        self.filter_size = args["filter_size"]
        self.num_filter = args["num_filter"]

        self.max_entities = args["max_entities"]
        self.entity_max_tokens = args["entity_max_tokens"]
        self.entity_max_chars = args["entity_max_chars"]

        # 인코더, 디코더 파라미터
        self.encoder_stack = args["encoder_stack"]
        self.encoder_max_step = args["encoder_max_step"]
        self.encoder_hidden = args["encoder_hidden"]
        self.decoder_hidden = args["decoder_hidden"]

        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)

        # 모델 입력단 초기화
        self._placeholder_init()

        # 모델과 함께 학습하며 finetune되는 단어 임베딩 테이블
        finetune_table = tf.get_variable(name="word_embedding_table_finetuning", initializer=self.word_emb_table,
                                         trainable=True, dtype=tf.float32)

        # 사전 학습 값 그대로 사용할 고정 단어 임베딩 테이블
        fix_table = tf.get_variable(name="word_embedding_table_fix", initializer=self.word_emb_table,
                                    trainable=False, dtype=tf.float32)
        # 임의 초기화 문자 임베딩 테이블
        char_emb_table = tf.get_variable("char_emb_table", shape=[self.char_vocab_size, self.char_emb_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 임의 초기화 개체 타입 임베딩 테이블
        entity_type_emb_table = tf.get_variable("entity_type_emb_table",
                                                shape=[self.entity_vocab_size, self.entity_type_emb_size],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 문장 인덱스 one-hot 임베딩 테이블
        sentence_id_emb_table = tf.eye(num_rows=self.max_sentences)

        # 문장 단어 임베딩
        context_embedding = self._context_embedding_layer(fix_table=fix_table, finetune_table=finetune_table,
                                                          char_emb_table=char_emb_table)
        # 문장 개체 임베딩
        entity_type_embedding = tf.nn.embedding_lookup(entity_type_emb_table, self.context_entity_type)
        # 문장 인덱스 임베딩
        sentence_id_embedding = tf.nn.embedding_lookup(sentence_id_emb_table, self.sentence_id)

        # entity token, character, type, position, sentence_id embedding
        entity_embedding = self._entity_pool_embedding(fix_table=fix_table, finetune_table=finetune_table,
                                                       char_emb_table=char_emb_table,
                                                       token_entities=self.entity_pool,
                                                       char_entities=self.char_entity_pool)

        # 문장에 있는 개체의 임베딩 가져오는 부분
        context_entity_emb = []
        unstack_entity_pool = tf.unstack(entity_embedding, axis=0)
        unstack_context_entity_id = tf.unstack(self.context_entity_id, axis=0)
        for entity_pool, context in zip(unstack_entity_pool, unstack_context_entity_id):
            context_entity_emb.append(tf.nn.embedding_lookup(entity_pool, context))

        context_entity_emb = tf.stack(context_entity_emb, axis=0)

        # context token, character, entity_type, sentence_id embedding
        context_embedding = tf.concat([context_embedding, entity_type_embedding, sentence_id_embedding, context_entity_emb],
                                      axis=-1)

        # 개체 임베딩, 개체 문장 인덱스 임베딩
        entity_pool_type_emb = tf.nn.embedding_lookup(entity_type_emb_table, self.entity_pool_type)
        entity_pool_sent_emb = tf.nn.embedding_lookup(sentence_id_emb_table, self.entity_sent_id)

        entity_pool_emb = tf.concat([entity_embedding, entity_pool_type_emb, entity_pool_sent_emb], axis=-1)

        # 관계 없는 개체가 포인팅하게 할 none 벡터
        none_emb = tf.get_variable(name="none_emb", shape=[self.decoder_hidden]
                                   , initializer=tf.zeros_initializer)
        pad_emb = tf.get_variable(name="pad_emb", shape=[self.decoder_hidden]
                                  , initializer=tf.zeros_initializer)

        pad_token = tf.expand_dims(tf.stack([pad_emb] * self.batch_size, 0), axis=1, name="pad_token")
        none_token = tf.expand_dims(tf.stack([none_emb] * self.batch_size, 0), axis=1, name="none_token")

        # 문장 인코딩
        encoder_output, encoder_state = self._biGRU_encoding_layer(encoder_input=context_embedding,
                                                                   encoder_length=self.context_input_length,
                                                                   name="encoder_layer")

        # 개체 인코딩 및 문장 개체 간 주의 집중
        pointing_mem, decoder_state = self._entity_encoding_layer(entity_pool_emb, encoder_output, encoder_state)

        # 디코더에서 포인팅 할 타겟
        self.pointing_target = tf.concat([pad_token, none_token, pointing_mem], axis=1)
        # 디코더 입력
        decoder_input = tf.concat([entity_pool_emb, pointing_mem], axis=-1)

        # 디코더 레이어 및 train op
        self._dual_pointer_decoder(decoder_input=decoder_input, decoder_init_state=decoder_state,
                                   decoder_hidden=self.decoder_hidden, pointing_memory=self.pointing_target)

    def _placeholder_init(self):
        '''
        모델 입력단 초기화 함수
        :return:
        '''
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

            self.entity_pool = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_entities, self.entity_max_tokens],
                                              name="entity_pool")
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

    def _context_embedding_layer(self, fix_table, finetune_table, char_emb_table):
        '''
        문장 내 단어들 임베딩하는 layer
        :param fix_table: 단어 임베딩 고정 테이블
        :param finetune_table: 단어 임베딩 학습 테이블
        :param char_emb_table: 문자 임베딩 테이블
        :return:
        context_emb: 임베딩 결과
        '''
        with tf.variable_scope(name_or_scope="context_embedding_layer"):
            _word_emb_fix = tf.nn.embedding_lookup(fix_table, self.context_input)
            _word_emb_finetune = tf.nn.embedding_lookup(finetune_table, self.context_input)

            char_emb = tf.nn.embedding_lookup(char_emb_table, self.char_context_input)
            char_token_emb = self._cnn_char_emb(char_emb, name="char_emb")
            context_emb = tf.concat([_word_emb_fix, _word_emb_finetune, char_token_emb], -1)

            return context_emb

    def _entity_pool_embedding(self, fix_table, finetune_table, char_emb_table, token_entities, char_entities):
        '''
        문장 내 개체들 임베딩하는 부분
        :param fix_table: 단어 임베딩 고정 테이블
        :param finetune_table: 단어 임베딩 학습 테이블
        :param char_emb_table: 문자 임베딩 테이블
        :param token_entities: 개체를 구성하는 단어
        :param char_entities: 개체를 구성하는 문자
        :return:
        entity_emb: 개체 임베딩 결과
        '''
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
        '''
        단어 단위 CNN embedding layer
        :param inputs: 임베딩 대상
        :param name: scope name
        :return:
        CNN 임베딩 결과
        '''
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
        문자 단위 CNN embedding layer
        :param embedding: 임베딩 대상
        :param name: scope name
        :return:
        '''
        emb_size = embedding.shape[-1]
        with tf.variable_scope(name):
            # 풀링 된 벡터 값들 저장할 리스트
            _pooled_outputs = []

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

            emb_output = tf.concat(_pooled_outputs, axis=2)  # [batch, step, convolution_embedding size]
        return emb_output

    def _biGRU_encoding_layer(self, encoder_input, encoder_length, name, num_stack=1):
        '''
        RNN으로 인코딩하는 계층, 양방향 GRU 사용
        :param
        encoder_input: 입력, [ batch, max_step, token_emb_size ]
        encoder_length: 입력 길이
        name: encoder scope name
        num_stack: stack 횟수
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
        '''
        multi-head attention
        :param key: key
        :param query: query
        :param value: value, self attention 일 시 key, query, value 다 같은 값
        :param attention_name: scope name
        :param num_heads: head 개수
        :param head_size: head size, 분할 후 차원 수
        :param intermediate_size: 마지막 FFN layer
        :param return_type: 어떤 식으로 결과를 낼 것인지 결정
        :return:
        '''
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

    def _entity_encoding_layer(self, entity_input, context_encoder_output):
        '''
        개체 인코딩, 문장 및 개체 간 주의집중 계층
        :param entity_input: 문장 내 개체 임베딩
        :param context_encoder_output: 문장 인코딩 결과
        :return:
        output: 개체 및 문장 주의 집중 결과
        state: entity encoder final state
        '''
        with tf.variable_scope("entity_encoding_layer"):
            sequence_len = [self.max_entities] * self.batch_size
            memory = context_encoder_output

            encoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.decoder_hidden,
                                                                 dropout_keep_prob=self.keep_pob)

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

    def _dual_pointer_decoder(self, decoder_input, decoder_init_state, decoder_hidden, pointing_memory):
        '''
        듀얼 포인터 네트워크 디코더 및 train operate layer
        :param decoder_input: 디코더 입력
        :param decoder_init_state: 디코더 초기 상태 값, 인코더 최종 state 사용
        :param decoder_hidden: 디코더 은닉층 사이즈
        :param pointing_memory: 디코더에서 포인팅 할 타겟
        :return:
        '''
        with tf.variable_scope("decoder_v3"):
            init_state = decoder_init_state

            with tf.variable_scope("object_cell_define"):
                object_decoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(decoder_hidden, dropout_keep_prob=self.keep_pob)
                object_cell_pre_state = init_state

            with tf.variable_scope("subject_cell_define"):
                subject_decoder_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(decoder_hidden, dropout_keep_prob=self.keep_pob)
                subject_cell_pre_state = init_state

            with tf.variable_scope("decoder_input_layer"):
                decoder_input_per_step = tf.unstack(decoder_input, axis=1)

            with tf.variable_scope("decoding_triple", reuse=tf.AUTO_REUSE):
                # 듀얼 포인팅 부분
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

                    # 포인팅은 multi-head attention 기반으로 수행
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
                # train operate 부분
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

                # Adam optimizer 및 EMA 사용, 학습 parameter tuning
                _optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self._gradients = _optimizer.compute_gradients(self.loss)
                # for g in self._gradients:
                #     print(g)
                _apply_op = _optimizer.apply_gradients(self._gradients, global_step=self.global_step)
                _ema = tf.train.ExponentialMovingAverage(decay=0.9999)

                with tf.control_dependencies([_apply_op]):
                    _ema_op = _ema.apply(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
                    self.train_op = tf.group(_ema_op)

                self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    def train_step(self, sess, inputs):
        '''
        학습 함수
        :param sess: tf.Sesstion
        :param inputs: placeholder input
        :return:
        train op 후 loss들 반환
        '''
        output = [self.object_loss, self.re_loss, self.subject_loss, self.rev_re_loss, self.loss, self.train_op]

        placeholders = [self.context_input, self.context_input_length, self.context_entity_type, self.context_entity_id,
                        self.sentence_id, self.char_context_input, self.entity_pool, self.entity_pool_type,
                        self.entity_sent_id, self.char_entity_pool, self.object_target,
                        self.relation_target, self.subject_target,self.rev_relation_target]

        input_feed = {holder: input for holder, input in zip(placeholders, inputs)}

        object_loss, re_loss, subject_loss, rev_loss, loss, _ = sess.run(output, feed_dict=input_feed)
        np.set_printoptions(threshold=np.nan, linewidth=350)

        return object_loss, re_loss, subject_loss, rev_loss, loss

    def valid_step(self, sess, inputs):
        '''
        development
        :param sess:
        :param inputs:
        :return:
        개발 셋 모델 예측 값
        '''
        output = [self.object_predicts, self.relation_predicts, self.subject_predicts, self.rev_relation_predicts,
                  self.object_target, self.relation_target]

        placeholders = [self.context_input, self.context_input_length, self.context_entity_type, self.context_entity_id,
                        self.sentence_id, self.char_context_input, self.entity_pool,
                        self.entity_pool_type, self.entity_sent_id, self.char_entity_pool,
                        self.object_target, self.relation_target]

        input_feed = {holder: input for holder, input in zip(placeholders, inputs)}

        object_predict, relation_predict, subject_predict, rev_re_predict, et, rt = sess.run(output, feed_dict=input_feed)
        np.set_printoptions(threshold=np.nan, linewidth=200)

        return object_predict, relation_predict, subject_predict, rev_re_predict

    def predict_step(self, sess, inputs):
        '''
        predict
        :param sess:
        :param inputs:
        :return:
        평가 셋 모델 예측 값
        '''
        output = [self.object_predicts, self.relation_predicts, self.subject_predicts, self.rev_relation_predicts]
        placeholders = [self.context_input, self.context_input_length, self.context_entity_type, self.context_entity_id,
                        self.sentence_id, self.char_context_input, self.entity_pool,
                        self.entity_pool_type, self.entity_sent_id, self.char_entity_pool]
        input_feed = {holder: input for holder, input in zip(placeholders, inputs)}
        object_predict, relation_predict, subject_predict, rev_relation_predict = sess.run(output, feed_dict=input_feed)

        return object_predict, relation_predict, subject_predict, rev_relation_predict


    def save_model(self, sess, path, epo_num):
        '''
        파라미터 저장 시 호출
        :param sess:
        :param path: 저장될 경로
        :param epo_num: 현재 epoch 수
        :return:
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, str(epo_num) + "_model.ckpt")
        print("Saving model, %s" % checkpoint_path)
        self.saver.save(sess, checkpoint_path)