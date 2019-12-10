import json
import numpy as np
import tensorflow as tf
from RE_transfer_version import RelationExtraction as re_model
import re, os, random
from tqdm import tqdm

# 모델의 하이퍼 파라미터들
args = {
    # True: 학습, False: 평가
    "is_train": False,
    "batch_size": 37,
    # 드랍아웃의 keep-probability, 학습 시에 0.9 사용했음
    "keep_prob": 1.0,
    "learning_rate": 0.001,

    # 문자 단위 random initialize 벡터 사이즈
    "char_emb_size": 50,
    # 최대 문장 갯수
    "max_sentences": 5,
    # 단어 최대 길이
    "word_maxlen": 25,
    # 단어 임베딩 벡터 사이즈
    "word_emb_size": 300,
    # 개체 타입 random initialize 벡터 사이즈
    "entity_type_emb_size": 50,

    # 문장 내 개체명 최대 개수
    "max_entities": 23,
    # 개체를 구성하는 단어 최대 개수
    "entity_max_tokens": 20,
    # 개체를 구성하는 문자 최대 개수
    "entity_max_chars": 80,

    # CNN 하이퍼 파라미터, 필터 사이즈 및 필터 개수
    "filter_size": [3, 4, 5],
    "num_filter": 100,

    # 문장 encoder stack 횟수
    "encoder_stack": 1,
    # encoder LSTM 최대 step, 문장 최대 길이
    "encoder_max_step": 90,
    # encoder, decoder hidden size
    "encoder_hidden": 256,
    "decoder_hidden": 512,

    # 사전 학습 단어 임베딩이 저장되어 있는 파일
    "embedding_file": "data/glove.840B.300d_inData_single.txt",
    # 학습, 개발, 평가 파일
    "train_file": "data/single_train_data.json",
    "develop_file": "data/single_valid_data.json",
    "test_file": "data/single_test_data.json",
    # 문자, 개체, 관계명 vocabulary
    "char_vocab": "data/single_char_vocab.txt",
    "entity_vocab": "data/single_entity_vocab.txt",
    "relation_vocab": "data/single_relation_vocab.txt",

    # 모델이 저장 될 위치
    "model_dir": "model/single/reverse_re_with_contextAttention(mh)_pointer(mh)",
    # 최대 학습 수
    "epoch": 1000,
    # 정기 save 주기
    "save_point": 100
}

def insert_padding(data, max_dim):
    '''
    padding 삽입 함수
    :param data: 패딩 삽입 할 데이터
    :param max_dim: 맞춰주고 싶은 차원 수, [r_0, r_1, ... , r_n] 꼴
    :return:
    '''
    def recurrent(data, pad_mat, depth):
        if depth != 0:
            for idx, _ in enumerate(pad_mat):
                if len(data) <= idx:
                    continue
                recurrent(data[idx], pad_mat[idx], depth - 1)
        else:
            for idx, _ in enumerate(pad_mat):
                if len(data) <= idx:
                    break
                pad_mat[idx] = data[idx]

    pad_mat = np.zeros(max_dim, dtype=np.int32)
    pad_mat = pad_mat.tolist()

    recurrent(data, pad_mat, len(max_dim) - 1)

    return pad_mat

def read_embedding():
    '''
    단어 임베딩, vocab 파일들을 읽고 사전(dict) 형태로 제작
    :return:
    embedding_table: 단어 임베딩 벡터들이 저장된 list
    idx2~, ~2idx_dic: 각 vocab의 단어들을 인덱스로 치환, 인덱스를 원래 단어로 바꿀 때 사용할 dict
    '''
    embedding_file = args["embedding_file"]
    char_vocab_file = args["char_vocab"]
    relation_vocab_file = args["relation_vocab"]

    word2idx_dic, idx2word_dic = {"<Padding>": 0, "<UNK>": 1}, {0: "<Padding>", 1: "<UNK>"}
    char2idx_dic, idx2char_dic = {"<Padding>": 0, "<UNK>": 1, " ": 2}, {0: "<Padding>", 1: "<UNK>", 2: " "}
    entity2idx_dic, idx2entity_dic = {"<Padding>": 0}, {0: "<Padding>"}
    rel2idx_dic, idx2rel_dic = {"<Padding>": 0, "<Other>": 1}, {0: "<Padding>", 1: "<Other>"}

    embedding_table = np.random.normal(scale=1.5, size=[2, args["word_emb_size"]])
    embedding_table = embedding_table.tolist()

    with open(embedding_file, "r", encoding="utf-8") as r_f:
        lines = r_f.readlines()

    for line in lines:
        elements = line.strip().split()
        word = elements[0]
        if len(elements) > 300:
            continue

        try:
            embedding = [float(fl) for fl in elements[1:]]
        except:
            print(elements)
        idx = len(word2idx_dic)

        word2idx_dic[word] = idx
        idx2word_dic[idx] = word
        embedding_table.append(embedding)

    with open(char_vocab_file, "r", encoding="utf-8") as r_f:
        lines = r_f.readlines()

    for ch in lines:
        ch = ch.strip()
        idx = len(char2idx_dic)
        char2idx_dic[ch] = idx
        idx2char_dic[idx] = ch

    with open(args["entity_vocab"], "r", encoding="utf-8") as r_f:
        lines = r_f.readlines()

    for entity in lines:
        entity = entity.strip()
        idx = len(entity2idx_dic)
        entity2idx_dic[entity] = idx
        idx2entity_dic[idx] = entity

    with open(relation_vocab_file, "r", encoding="utf-8") as r_f:
        lines = r_f.readlines()

    for rel in lines:
        rel = rel.strip()
        idx = len(rel2idx_dic)
        rel2idx_dic[rel] = idx
        idx2rel_dic[idx] = rel

    args["char_vocab_size"] = len(char2idx_dic)
    args["entity_vocab_size"] = len(entity2idx_dic)
    args["relation_vocab_size"] = len(rel2idx_dic)

    return embedding_table, word2idx_dic, idx2word_dic, char2idx_dic, idx2char_dic, entity2idx_dic, idx2entity_dic, \
           rel2idx_dic, idx2rel_dic

def read_data(word2idx_dic, char2idx_dic, entity2idx_dic, rel2idx_dic, file_name):
    '''
    데이터 파일 읽고 모델 입력 형식에 맞게 바꿔주는 함수
    :param word2idx_dic: 단어->index 치환 dict
    :param char2idx_dic: 문자->index 치환 dict
    :param entity2idx_dic: 개체타입->index 치환 dict
    :param rel2idx_dic: 관계명->index 치환 dict
    :param file_name: 읽을 파일 위치
    :return:
    all_data: 모델 입력 형식대로 변환된 데이터 dict
    '''
    with open(file_name, "r", encoding="utf-8") as r_f:
        json_data = json.load(r_f)

    sp = ".,!?"
    all_data = []
    max_char_len = 0

    for paragraph in json_data:
        context = paragraph["context"]
        context_entity = paragraph["context_entity"]
        context_entity_id = paragraph["context_entity_id"]
        sentence_id = paragraph["sentence_id"]
        relations = paragraph["relations"]
        entity_pool = paragraph["entity_pool"]

        if len(context_entity_id) != len(sentence_id):
            print(len(context_entity_id))
            print(len(sentence_id))

            print(context)
            print(context_entity)
            print(context_entity_id)
            print(sentence_id)

        data = {}

        data["context"] = context
        context2idx = []
        context_tokens = context.split()
        context_length = len(context_tokens)
        context_char = []

        for token in context_tokens:
            if len(token) > max_char_len:
                max_char_len = len(token)

            if token[-1] in sp:
                token = token[:-1]
            token = token.lower()

            if token in word2idx_dic:
                context2idx.append(word2idx_dic[token])
            else:
                context2idx.append(word2idx_dic["<UNK>"])

            token_char = []
            for ch in token:
                if ch in char2idx_dic:
                    token_char.append(char2idx_dic[ch])
                else:
                    token_char.append(char2idx_dic["<UNK>"])
            context_char.append(token_char)

        data["context2id"] = insert_padding(context2idx, [args["encoder_max_step"]])
        data["context_length"] = context_length

        data["context_char2idx"] = insert_padding(context_char, [args["encoder_max_step"], args["word_maxlen"]])

        ce2idx = []
        for ce in context_entity:
            ce2idx.append(entity2idx_dic[ce])

        data["context_entity"] = insert_padding(ce2idx, [args["encoder_max_step"]])
        data["context_entity_id"] = insert_padding(context_entity_id, [args["encoder_max_step"]])
        data["sentence_id"] = insert_padding(sentence_id, [args["encoder_max_step"]])

        entity_pool_word2idx = []
        entity_pool_char2idx = []
        entity_pool_type2idx = []
        entity_sent_id = []

        for entity in entity_pool:
            entity_tokens = entity["entity"].split()
            entity_word2idx = []
            entity_char2idx = []
            for token in entity_tokens:
                if token[-1] in sp:
                    token = token[:-1]
                token = token.lower()

                if token in word2idx_dic:
                    entity_word2idx.append(word2idx_dic[token])
                else:
                    entity_word2idx.append(word2idx_dic["<UNK>"])

            for ch in entity["entity"]:
                if ch in char2idx_dic:
                    entity_char2idx.append(char2idx_dic[ch])
                else:
                    entity_char2idx.append(char2idx_dic["<UNK>"])

            entity_pool_word2idx.append(entity_word2idx)
            entity_pool_type2idx.append(entity2idx_dic[entity["type"]])
            entity_pool_char2idx.append(entity_char2idx)
            entity_sent_id.append(entity["sentence_id"])

        data["entity_pool_size"] = len(entity_pool)
        data["entity_pool"] = entity_pool

        data["entity_pool2idx"] = insert_padding(entity_pool_word2idx, [args["max_entities"], args["entity_max_tokens"]])
        data["entity_pool_type"] = insert_padding(entity_pool_type2idx, [args["max_entities"]])
        data["entity_sentence_id"] = insert_padding(entity_sent_id, [args["max_entities"]])
        data["entity_pool_char2idx"] = insert_padding(entity_pool_char2idx, [args["max_entities"], args["entity_max_chars"]])

        triples = set()

        d3_object = [0 for _ in range(args["max_entities"])]
        d3_relation = [0 for _ in range(args["max_entities"])]
        d3_subject = [0 for _ in range(args["max_entities"])]
        d3_rev_relation = [0 for _ in range(args["max_entities"])]

        for i, _ in enumerate(entity_pool):
            d3_object[i] = 1
            d3_relation[i] = 1
            d3_subject[i] = 1
            d3_rev_relation[i] = 1

        for relation in relations:
            d3_object[relation["subject_id"]-2] = relation["object_id"]
            d3_relation[relation["subject_id"]-2] = rel2idx_dic[relation["relation"]]
            d3_subject[relation["object_id"]-2] = relation["subject_id"]
            d3_rev_relation[relation["object_id"]-2] = rel2idx_dic[relation["relation"]]

            triples.add((relation["subject_id"], rel2idx_dic[relation["relation"]], relation["object_id"]))

        data["decoder3_object"] = d3_object
        data["decoder3_relation"] = d3_relation
        data["decoder3_subject"] = d3_subject
        data["decoder3_rev_relation"] = d3_rev_relation

        data["triples"] = list(triples)

        all_data.append(data)

    return all_data

def create_model(sess):
    '''
    모델을 새로 학습하거나 이전 모델 불러오는 함수
    :param sess: tf.Sesstion
    :return:
    model: 새로 만들어지거나 이전 모델을 불러온 결과
    '''
    # 그래프를 만듦
    model = re_model(args)
    save_path = os.path.join("./", args["model_dir"])
    # 체크포인트가 존재하면 해당 파일 위치 불러옴
    ckpt = tf.train.get_checkpoint_state(save_path)
    if ckpt:
        file_name = ckpt.model_checkpoint_path + ".meta"

    if ckpt and tf.gfile.Exists(file_name):
        # 불러온 체크포인트 모델 가중치를 현재 그래프에 덮어씌움
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        print("Success Load!")
    else:
        # 저장된 모델 없으면 새로운 가중치로 그래프 초기화
        print("Created model with fresh parameters.")
        sess.run(tf.global_variables_initializer())
    return model

def get_batch_data(batch_data):
    '''
    형식이 갖춰진 데이터를 batch 형태로 바꿔주는 함수 (data transpose)
    :param batch_data: batch 형태로 바꿔줄 데이터
    :return:
    형태가 바뀐 데이터들 반환
    '''
    context2idx = []
    context_length = []
    context_char2idx = []
    context_entity = []
    context_entity_id = []
    sentence_id = []

    entity_pool2idx = []
    entity_pool_type = []
    entity_sent_id = []
    entity_pool_char2idx = []

    object = []
    relation = []
    subject = []
    rev_relation = []

    for data in batch_data:
        context2idx.append(data["context2id"])
        context_length.append(data["context_length"])
        context_char2idx.append(data["context_char2idx"])
        context_entity.append(data["context_entity"])
        context_entity_id.append(data["context_entity_id"])
        sentence_id.append(data["sentence_id"])

        entity_pool2idx.append(data["entity_pool2idx"])
        entity_pool_type.append(data["entity_pool_type"])
        entity_sent_id.append(data["entity_sentence_id"])
        entity_pool_char2idx.append(data["entity_pool_char2idx"])

        object.append(data["decoder3_object"])
        relation.append(data["decoder3_relation"])
        subject.append(data["decoder3_subject"])
        rev_relation.append(data["decoder3_rev_relation"])

    return context2idx, context_length, context_entity, context_entity_id, sentence_id, context_char2idx, \
           entity_pool2idx, entity_pool_type, entity_sent_id, \
           entity_pool_char2idx, object, relation, subject, rev_relation

def get_batch_idex(data):
    '''
    지정된 batch_size만큼 모델에 입력되도록 데이터 개수를 계산해줌
    :param data: 모델 입력 데이터
    :return:
    index_list: batch_size만큼 분할된 데이터 인덱스들이 저장된 list
    '''
    batch_data_len = int(len(data) / args["batch_size"])
    index_list = []

    for i in range(batch_data_len):
        index_list.append((i * args["batch_size"], args["batch_size"] + i * args["batch_size"]))

    return index_list


def train():
    '''
    학습 함수
    '''
    # vocab 파일, embedding 파일, 데이터 파일 읽는 부분
    embedding_table, word2idx_dic, idx2word_dic, char2idx_dic, idx2char_dic, entity2idx_dic, idx2entity_dic, \
    rel2idx_dic, idx2rel_dic = read_embedding()
    args["embedding_table"] = embedding_table

    train_data = read_data(word2idx_dic, char2idx_dic, entity2idx_dic, rel2idx_dic, args["train_file"])
    valid_data = read_data(word2idx_dic, char2idx_dic, entity2idx_dic, rel2idx_dic, args["develop_file"])
    close_data = train_data[:500]

    print(len(train_data))
    print(len(valid_data))

    idx_list = get_batch_idex(train_data)

    save_path = os.path.join("./", args["model_dir"])

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        # 모델 불러옴
        model = create_model(sess)
        # 지정된 횟수(epoch)만큼 학습 진행
        for epoch in range(args["epoch"]):
            for idx in tqdm(range(len(idx_list))):
                start, end = idx_list[idx]

                object_loss, re_loss, subject_loss, rev_loss, loss = model.train_step(sess, get_batch_data(train_data[start:end]))
                if idx % 30 == 0:
                    print(format("\tEpoch %d (%d/%d)" % (epoch + 1, (idx + 1), len(idx_list))))
                    print("\nObject loss: ", object_loss, "\nRelarion loss: ", re_loss, "\nSubject loss: ",
                          subject_loss, "\nReverse relation loss: ", rev_loss, "\nTotal loss: ", loss)

            if (epoch + 1) % args["save_point"] == 0:
                # 정기적인 저장 주기가 되면 저장
                model.save_model(sess, save_path, epoch+1)

            if (epoch + 1) % 1 == 0:
                # valid, close test
                print("########## valid test ############")
                valid_idx_list = get_batch_idex(valid_data)
                object_pred_list, re_pred_list, subject_pred_list, rev_relation_pred_list = [], [], [], []

                for idx in tqdm(range(len(valid_idx_list))):
                    start, end = idx_list[idx]
                    batch_data = get_batch_data(valid_data[start:end])

                    inputs = batch_data

                    object_pred, re_pred, subject_pred, rev_re_pred = model.valid_step(sess, inputs)

                    for e in object_pred:
                        object_pred_list.append(e)
                    for e in re_pred:
                        re_pred_list.append(e)
                    for e in subject_pred:
                        subject_pred_list.append(e)
                    for e in rev_re_pred:
                        rev_relation_pred_list.append(e)

                valid_performance(valid_data, object_pred_list, re_pred_list, subject_pred_list, rev_relation_pred_list,
                                  idx2rel_dic)

                print("########## close test ############")
                valid_idx_list = get_batch_idex(close_data)
                object_pred_list, re_pred_list, subject_pred_list, rev_relation_pred_list = [], [], [], []

                for idx in tqdm(range(len(valid_idx_list))):
                    start, end = idx_list[idx]
                    batch_data = get_batch_data(close_data[start:end])

                    inputs = batch_data

                    object_pred, re_pred, subject_pred, rev_re_pred = model.valid_step(sess, inputs)

                    for e in object_pred:
                        object_pred_list.append(e)
                    for e in re_pred:
                        re_pred_list.append(e)
                    for e in subject_pred:
                        subject_pred_list.append(e)
                    for e in rev_re_pred:
                        rev_relation_pred_list.append(e)

                valid_performance(close_data, object_pred_list, re_pred_list, subject_pred_list, rev_relation_pred_list,
                                  idx2rel_dic)

            random.shuffle(train_data)
            idx_list = get_batch_idex(train_data)

def test():
    '''
    평가 함수
    '''
    embedding_table, word2idx_dic, idx2word_dic, char2idx_dic, idx2char_dic, entity2idx_dic, idx2entity_dic, \
    rel2idx_dic, idx2rel_dic = read_embedding()
    args["embedding_table"] = embedding_table

    test_data = read_data(word2idx_dic, char2idx_dic, entity2idx_dic, rel2idx_dic, args["test_file"])

    idx_list = get_batch_idex(test_data)
    print(len(test_data))

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        model = create_model(sess)
        print("########## test ############")
        valid_idx_list = get_batch_idex(test_data)
        object_pred_list, re_pred_list, subject_pred_list, rev_relation_pred_list = [], [], [], []

        for idx in tqdm(range(len(valid_idx_list))):
            start, end = idx_list[idx]
            batch_data = get_batch_data(test_data[start:end])

            inputs = batch_data[:-4]

            object_pred, re_pred, subject_pred, rev_re_pred = model.predict_step(sess, inputs)

            for e in object_pred:
                object_pred_list.append(e)
            for e in re_pred:
                re_pred_list.append(e)
            for e in subject_pred:
                subject_pred_list.append(e)
            for e in rev_re_pred:
                rev_relation_pred_list.append(e)

        macro_performance(test_data, object_pred_list, re_pred_list, subject_pred_list, rev_relation_pred_list,
                          idx2rel_dic)

def eval_f1(correct_cnt, recall_cnt, precision_cnt):
    '''
    f1 계산 함수
    :param correct_cnt: True positive
    :param recall_cnt: True positive + False negative
    :param precision_cnt: True positive + False positive
    :return:
    '''
    if recall_cnt == 0:
        recall = 0
    else:
        recall = correct_cnt / float(recall_cnt)

    if precision_cnt == 0:
        precision = 0
    else:
        precision = correct_cnt / float(precision_cnt)
    if (recall + precision == 0):
        f_1 = 0
    else:
        f_1 = (2 * recall * precision) / (recall + precision)
    return recall, precision, f_1

def valid_performance(valid_set, object_predicts, relation_predicts, subject_predicts, rev_relation_predicts,
                      idx2rel_dict):
    '''
    성능 측정 함수
    :param valid_set: 정답 데이터
    :param object_predicts: 모델이 예측한 object
    :param relation_predicts: 모델이 예측한 relation (sub->ob)
    :param subject_predicts: 모델이 예측한 subject
    :param rev_relation_predicts: 모델이 예측한 reverse relation (ob->sub)
    :param idx2rel_dict: 인덱스로 표현된 관계명을 다시 관계명으로 바꿔 줄 dict
    :return:
    '''
    ignore = [0, 1]
    recall_cnt, precision_cnt, correct_cnt = 0, 0, 0
    other_recall_cnt, other_precision_cnt, other_correct_cnt = 0, 0, 0
    rel_dic = {}

    for idx, (objects, relations, subjects, rev_relations) in enumerate(zip(object_predicts, relation_predicts,
                                                                            subject_predicts, rev_relation_predicts)):
        triple_targets = valid_set[idx]["triples"]
        triple_predicts = set()
        entity_pool_len = valid_set[idx]["entity_pool_size"]

        for sub, (ob, re) in enumerate(zip(objects, relations)):
            if sub > entity_pool_len:
                break
            sub_idx = sub + 2
            if (ob not in ignore) and (re not in ignore):
                triple_predicts.add((sub_idx, re, ob))

        for ob, (sub, rev_re) in enumerate(zip(subjects, rev_relations)):
            if ob > entity_pool_len:
                break
            ob_idx = ob + 2
            if (sub not in ignore) and (rev_re not in ignore):
                triple_predicts.add((sub, rev_re, ob_idx))

        triple_predicts = list(triple_predicts)

        triple_targets_with_other = make_triple_include_other(entity_pool_len, triple_targets)
        triple_predicts_with_other = make_triple_include_other(entity_pool_len, triple_predicts)

        for triple in triple_targets_with_other:
            if triple[1] in rel_dic:
                rel_dic[triple[1]] += 1
            else:
                rel_dic[triple[1]] = 1

        for pre_triple in triple_predicts:
            if pre_triple in triple_targets:
                correct_cnt += 1

        for pre_triple in triple_predicts_with_other:
            if pre_triple in triple_targets_with_other:
                other_correct_cnt += 1

        precision_cnt += len(triple_predicts)
        recall_cnt += len(triple_targets)

        other_precision_cnt += len(triple_predicts_with_other)
        other_recall_cnt += len(triple_targets_with_other)

        print("context: ", valid_set[idx]["context"])
        print("targets:", triple_targets)
        triple_index2word(triple_targets, valid_set[idx]["entity_pool"], idx2rel_dict)
        print("predicts:", triple_predicts)
        triple_index2word(triple_predicts, valid_set[idx]["entity_pool"], idx2rel_dict)
        print("==================================================")

    print("dst:", rel_dic)

    print("F1 score of triple fact")
    recall, precision, f_1 = eval_f1(correct_cnt, recall_cnt, precision_cnt)
    print("recall:", recall)
    print("precision:", precision)
    print("F1:", f_1)

    print("F1 score of triple fact with other")
    recall, precision, f_1 = eval_f1(other_correct_cnt, other_recall_cnt, other_precision_cnt)
    print("recall:", recall)
    print("precision:", precision)
    print("F1:", f_1)

def triple_index2word(triple_list, entity_pool, idx2rel_dict):
    '''
    문장의 인덱스로 표현된 트리플을 문자열로 복원하는 함수
    :param triple_list: index로 표현된 트리플
    :param entity_pool: 문장의 개체들
    :param idx2rel_dict: 인덱스로 표현된 관계명을 다시 관계명으로 바꿔 줄 dict
    :return:
    '''
    for triple in triple_list:
        sub_id, rel_id, ob_id = triple
        subject, object = "None", "None"
        subject_type, object_type = "None", "None"

        for entity in entity_pool:
            if entity["id"] == sub_id:
                subject = entity["entity"]
                subject_type = entity["type"]
            if entity["id"] == ob_id:
                object = entity["entity"]
                object_type = entity["type"]

        relation = idx2rel_dict[rel_id]

        print("\tsubject:", subject, "\ttype:", subject_type)
        print("\trelation:", relation)
        print("\tobject:", object, "\ttype:", object_type)

def make_triple_include_other(entity_pool_len, default_triple: list):
    '''
    아무 관계를 갖지 않는 개체도 트리플로 표현하기 위한 함수
    :param entity_pool_len: 문장의 전체 개체 수
    :param default_triple: 실제로 관계를 갖고 있는 개체들이 포함된 트리플
    :return:
    '''
    triple_include_other = set()

    for sub_id in range(entity_pool_len):
        sub_id = sub_id + 2
        flag = False

        for triple in default_triple:
            if triple[0] == sub_id:
                flag = True
                triple_include_other.add(triple)

        if not flag:
            triple_include_other.add((sub_id, 1, 1))

    return list(triple_include_other)

def macro_performance(valid_set, object_predicts, relation_predicts, subject_predicts, rev_relation_predicts,
                      idx2rel_dict):
    '''
    각 관계명 별로 성능 측정하기 위한 함수
    :param valid_set: 정답 데이터
    :param object_predicts: 모델이 예측한 object
    :param relation_predicts: 모델이 예측한 relation (sub->ob)
    :param subject_predicts: 모델이 예측한 subject
    :param rev_relation_predicts: 모델이 예측한 reverse relation (ob->sub)
    :param idx2rel_dict: 인덱스로 표현된 관계명을 다시 관계명으로 바꿔 줄 dict
    :return:
    '''
    ignore = [0, 1]
    recall_cnt, precision_cnt, correct_cnt = 0, 0, 0
    other_recall_cnt, other_precision_cnt, other_correct_cnt = 0, 0, 0
    rel_dic = {}

    macro_re, macro_pr, macro_cr = {}, {}, {}

    for idx, (objects, relations, subjects, rev_relations) in enumerate(zip(object_predicts, relation_predicts,
                                                                            subject_predicts, rev_relation_predicts)):
        triple_targets = valid_set[idx]["triples"]
        triple_predicts = set()
        entity_pool_len = valid_set[idx]["entity_pool_size"]

        # 문장 내 개체 개수 별로 성능 측정하고 싶을 때 사용
        # if entity_pool_len < 5:
        #     continue

        for sub, (ob, re) in enumerate(zip(objects, relations)):
            if sub > entity_pool_len:
                continue
            sub_idx = sub + 2
            if (ob not in ignore) and (re not in ignore):
                triple_predicts.add((sub_idx, re, ob))

        for ob, (sub, rev_re) in enumerate(zip(subjects, rev_relations)):
            if ob > entity_pool_len:
                continue
            ob_idx = ob + 2
            if (sub not in ignore) and (rev_re not in ignore):
                triple_predicts.add((sub, rev_re, ob_idx))

        triple_predicts = list(triple_predicts)

        triple_targets_with_other = make_triple_include_other(entity_pool_len, triple_targets)
        triple_predicts_with_other = make_triple_include_other(entity_pool_len, triple_predicts)

        for triple in triple_targets_with_other:
            if triple[1] in rel_dic:
                rel_dic[triple[1]] += 1
            else:
                rel_dic[triple[1]] = 1

        for pre_triple in triple_predicts:
            if pre_triple in triple_targets:
                correct_cnt += 1

        for pre_triple in triple_predicts_with_other:
            relation_idx = pre_triple[1]
            if pre_triple in triple_targets_with_other:
                other_correct_cnt += 1

                if relation_idx in macro_cr:
                    macro_cr[relation_idx] += 1
                else:
                    macro_cr[relation_idx] = 1
            if relation_idx in macro_pr:
                macro_pr[relation_idx] += 1
            else:
                macro_pr[relation_idx] = 1

        precision_cnt += len(triple_predicts)
        recall_cnt += len(triple_targets)

        other_precision_cnt += len(triple_predicts_with_other)
        other_recall_cnt += len(triple_targets_with_other)

        print("context: ", valid_set[idx]["context"])
        print("targets:", triple_targets)
        triple_index2word(triple_targets, valid_set[idx]["entity_pool"], idx2rel_dict)
        print("predicts:", triple_predicts)
        triple_index2word(triple_predicts, valid_set[idx]["entity_pool"], idx2rel_dict)
        print("==================================================")

    print(rel_dic)
    print(macro_pr)
    print(macro_cr)
    print(other_recall_cnt)
    print(other_precision_cnt)
    print(other_correct_cnt)

    for k in rel_dic:
        relation = idx2rel_dict[k]
        mre = macro_cr[k]/float(rel_dic[k])
        mpc = macro_cr[k]/float(macro_pr[k])
        mf1 = (2*mre*mpc)/(mre+mpc)
        print(relation, "macro f1:", mf1)
        print("===============================")

    recall, precision, f_1 = eval_f1(correct_cnt, recall_cnt, precision_cnt)
    print("Recall: ", recall)
    print("Precision:", precision)
    print("F1:", f_1)

    recall, precision, f_1 = eval_f1(other_correct_cnt, other_recall_cnt, other_precision_cnt)
    print("other_Recall: ", recall)
    print("other_Precision:", precision)
    print("other_F1:", f_1)

if __name__=="__main__":
    if args["is_train"]:
        print("train model...")
        train()
    else:
        print("test...")
        test()
