import json
import numpy as np
import tensorflow as tf
from Relation_extration import RelationExtraction as re_model
import re, os, random
from tqdm import tqdm

args = {
    "is_train": False,
    "batch_size": 37,
    "keep_prob": 1.0,
    "learning_rate": 0.001,

    "char_emb_size": 50,
    "charemb_rnn_hidden": 64,
    "tokenemb_rnn_hidden": 128,
    "max_sentences": 5,
    "word_maxlen": 25,
    "word_emb_size": 300,
    "sentence_emb_size": 100,
    "position_emb_size": 100,
    "entity_emb_size": 50,

    "max_entities": 23,
    "entity_max_tokens": 20,
    "entity_max_chars": 80,

    "filter_size": [3, 4, 5],
    "num_filter": 100,

    "encoder_stack": 1,
    "encoder_max_step": 90,
    "encoder_hidden": 256,
    "decoder_hidden": 512,
    "max_relations": 12,
    "max_relation_entities": 20,
    "attention_hidden": 256,

    "embedding_file": "data/glove.840B.300d_inData_single.txt",
    "train_file": "data/single_train_data.json",
    "develop_file": "data/single_valid_data.json",
    "test_file": "data/single_test_data.json",
    "char_vocab": "data/single_char_vocab.txt",
    "entity_vocab": "data/single_entity_vocab.txt",
    "relation_vocab": "data/single_relation_vocab.txt",

    "model_dir": "model/single/reverse_re_with_contextAttention(mh)_pointer(mh)",
    "epoch": 1000,
    "save_point": 100
}

def read_embedding():
    embedding_file = args["embedding_file"]
    char_vocab_file = args["char_vocab"]
    relation_vocab_file = args["relation_vocab"]

    word2idx_dic, idx2word_dic = {"<Padding>": 0, "<UNK>": 1}, {0: "<Padding>", 1: "<UNK>"}
    char2idx_dic, idx2char_dic = {"<Padding>": 0, "<UNK>": 1, " ": 2}, {0: "<Padding>", 1: "<UNK>", 2: " "}
    entity2idx_dic, idx2entity_dic = {"<Padding>": 0}, {0: "<Padding>"}
    rel2idx_dic, idx2rel_dic = {"<Padding>": 0, "<End>": 1}, {0: "<Padding>", 1: "<End>"}

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

        context_entity_id_list = paragraph["context_entity_id_list"]

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
        context_char_length = []

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
            token_len = len(token)
            # if len(token) > 20:
            #     print(token, len(token))
            for ch in token[:args["word_maxlen"]]:
                if ch in char2idx_dic:
                    token_char.append(char2idx_dic[ch])
                else:
                    token_char.append(char2idx_dic["<UNK>"])
            pad_list = [0 for _ in range(args["word_maxlen"] - token_len)]
            context_char.append(token_char + pad_list)
            context_char_length.append(token_len)

        pad_list = [0 for _ in range(args["encoder_max_step"] - context_length)]
        data["context2id"] = context2idx + pad_list
        data["context_length"] = context_length

        pad_list = [[0 for _ in range(args["word_maxlen"])] for _ in range(args["encoder_max_step"] - context_length)]
        data["context_char2idx"] = context_char + pad_list
        pad_list = [0 for _ in range(args["encoder_max_step"] - context_length)]
        data["context_chat_length"] = context_char_length + pad_list

        ce2idx = []
        for ce in context_entity:
            ce2idx.append(entity2idx_dic[ce])

        pad_list = [0 for _ in range(args["encoder_max_step"] - context_length)]
        data["context_entity"] = ce2idx + pad_list
        data["context_entity_id"] = context_entity_id + pad_list
        data["sentence_id"] = sentence_id + pad_list

        data["context_entity_id_list"] = context_entity_id_list

        ep2idx = []
        entity_pool_length = []
        ep2idx_char = []
        entity_pool_charlen = []
        ep_type2idx = []
        entity_sent_id = []

        for entity in entity_pool:
            entity_tokens = entity["entity"].split()
            et2idx = []
            et2idx_char = []
            for token in entity_tokens[:args["entity_max_tokens"]]:
                if token[-1] in sp:
                    token = token[:-1]
                token = token.lower()

                if token in word2idx_dic:
                    et2idx.append(word2idx_dic[token])
                else:
                    et2idx.append(word2idx_dic["<UNK>"])

            for ch in entity["entity"][:args["entity_max_chars"]]:
                if ch in char2idx_dic:
                    et2idx_char.append(char2idx_dic[ch])
                else:
                    et2idx_char.append(char2idx_dic["<UNK>"])

            pad_list = [0 for _ in range(args["entity_max_tokens"]-len(et2idx))]
            ep2idx.append(et2idx + pad_list)
            entity_pool_length.append(len(et2idx))

            ep_type2idx.append(entity2idx_dic[entity["type"]])

            pad_list = [0 for _ in range(args["entity_max_chars"] - len(et2idx_char))]
            ep2idx_char.append(et2idx_char + pad_list)
            entity_pool_charlen.append(len(et2idx_char))

            entity_sent_id.append(entity["sentence_id"])

        data["entity_pool_size"] = len(entity_pool)
        data["entity_pool"] = entity_pool

        pad_list = [[0 for _ in range(args["entity_max_tokens"])] for _ in range(args["max_entities"]-len(ep2idx))]
        data["entity_pool2idx"] = ep2idx + pad_list
        pad_list = [0 for _ in range(args["max_entities"]-len(ep2idx))]
        data["entity_pool_length"] = entity_pool_length + pad_list
        data["entity_pool_type"] = ep_type2idx + pad_list
        data["entity_sentence_id"] = entity_sent_id + pad_list

        pad_list = [[0 for _ in range(args["entity_max_chars"])] for _ in range(args["max_entities"] - len(ep2idx))]
        data["entity_pool_char2idx"] = ep2idx_char + pad_list
        pad_list = [0 for _ in range(args["max_entities"]-len(ep2idx))]
        data["entity_pool_charlen"] = entity_pool_charlen + pad_list

        subjects = []
        objects = []
        relation_types = []

        entities = []
        triples = set()

        d3_object = [0 for _ in range(args["max_entities"])]
        d3_relation = [0 for _ in range(args["max_entities"])]
        d3_subject = [0 for _ in range(args["max_entities"])]
        d3_rev_relation = [0 for _ in range(args["max_entities"])]

        d4_entity = [0 for _ in range(args["encoder_max_step"])]
        d4_relation = [0 for _ in range(args["encoder_max_step"])]

        for i, _ in enumerate(entity_pool):
            d3_object[i] = 1
            d3_relation[i] = 1
            d3_subject[i] = 1
            d3_rev_relation[i] = 1

        for i in range(context_length):
            d4_entity[i] = 1
            d4_relation[i] = 1

        entity_pair = []
        nested_relation = []

        # if context == "your mom and dad":
        #     print(relations)

        for relation in relations:
            subjects.append(relation["subject_id"])
            objects.append(relation["object_id"])
            entity_pair.append((relation["subject_id"], relation["object_id"]))

            entities.append(relation["subject_id"])
            entities.append(relation["object_id"])
            relation_types.append(rel2idx_dic[relation["relation"]])

            d3_object[relation["subject_id"]-2] = relation["object_id"]
            d3_relation[relation["subject_id"]-2] = rel2idx_dic[relation["relation"]]
            d3_subject[relation["object_id"]-2] = relation["subject_id"]
            d3_rev_relation[relation["object_id"]-2] = rel2idx_dic[relation["relation"]]

            d4_entity[relation["subject_info"]["word_idx"][-1]-1] = relation["object_info"]["word_idx"][0]+1
            d4_relation[relation["subject_info"]["word_idx"][-1]-1] = rel2idx_dic[relation["relation"]]

            triples.add((relation["subject_id"], rel2idx_dic[relation["relation"]], relation["object_id"]))

        end_symbol = 1
        subjects.append(end_symbol)
        objects.append(end_symbol)
        entities.append(end_symbol)
        relation_types.append(rel2idx_dic["<End>"])

        # pad_list = [(args["max_entities"]-1) for _ in range(args["max_relations"]-len(subjects))]
        pad_list = [0 for _ in range(args["max_relations"] - len(subjects))]
        subjects += pad_list
        objects += pad_list
        pad_list = [0 for _ in range(args["max_relations"]-len(relation_types))]
        relation_types += pad_list
        pad_list = [0 for _ in range(args["max_relation_entities"] - len(entities))]
        entities += pad_list

        # print(context)
        # print(d3_entity)
        # print(d3_relation)
        # print(triples)
        # print()

        data["subject"] = subjects
        data["object"] = objects
        data["relation"] = relation_types
        data["relation_entities"] = entities

        data["decoder3_object"] = d3_object
        data["decoder3_relation"] = d3_relation
        data["decoder3_subject"] = d3_subject
        data["decoder3_rev_relation"] = d3_rev_relation

        data["triples"] = list(triples)

        all_data.append(data)

    return all_data

def create_model(sess):
    model = re_model(args)
    save_path = os.path.join("./", args["model_dir"])
    ckpt = tf.train.get_checkpoint_state(save_path)
    if ckpt:
        file_name = ckpt.model_checkpoint_path + ".meta"

    if ckpt and tf.gfile.Exists(file_name):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        print("Success Load!")
    else:
        print("Created model with fresh parameters.")
        sess.run(tf.global_variables_initializer())
    return model

def get_batch_data(batch_data):
    context2idx = []
    context_length = []
    context_char2idx = []
    context_chat_length = []
    context_entity = []
    context_entity_id = []
    sentence_id = []

    entity_pool2idx = []
    entity_pool_length = []
    entity_pool_type = []
    entity_sent_id = []
    entity_pool_char2idx = []
    entity_pool_charlen = []

    object = []
    relation = []
    subject = []
    rev_relation = []

    for data in batch_data:
        context2idx.append(data["context2id"])
        context_length.append(data["context_length"])
        context_char2idx.append(data["context_char2idx"])
        context_chat_length.append(data["context_chat_length"])
        context_entity.append(data["context_entity"])
        context_entity_id.append(data["context_entity_id"])
        sentence_id.append(data["sentence_id"])

        entity_pool2idx.append(data["entity_pool2idx"])
        entity_pool_length.append(data["entity_pool_length"])
        entity_pool_type.append(data["entity_pool_type"])
        entity_sent_id.append(data["entity_sentence_id"])
        entity_pool_char2idx.append(data["entity_pool_char2idx"])
        entity_pool_charlen.append(data["entity_pool_charlen"])

        object.append(data["decoder3_object"])
        relation.append(data["decoder3_relation"])
        subject.append(data["decoder3_subject"])
        rev_relation.append(data["decoder3_rev_relation"])

        # print()
        # print(data["decoder3_object"])
        # print(data["decoder3_relation"])
        # print(data["decoder3_subject"])
        # print(data["decoder3_rev_relation"])

    return context2idx, context_length, context_entity, context_entity_id, sentence_id, context_char2idx, \
           context_chat_length, entity_pool2idx, entity_pool_length, entity_pool_type, entity_sent_id, \
           entity_pool_char2idx, entity_pool_charlen, object, relation, subject, rev_relation

def get_batch_idex(data):
    batch_data_len = int(len(data) / args["batch_size"])
    index_list = []

    for i in range(batch_data_len):
        index_list.append((i * args["batch_size"], args["batch_size"] + i * args["batch_size"]))

    return index_list


def train():
    embedding_table, word2idx_dic, idx2word_dic, char2idx_dic, idx2char_dic, entity2idx_dic, idx2entity_dic, \
    rel2idx_dic, idx2rel_dic = read_embedding()
    args["embedding_table"] = embedding_table

    train_data = read_data(word2idx_dic, char2idx_dic, entity2idx_dic, rel2idx_dic, args["train_file"])
    valid_data = read_data(word2idx_dic, char2idx_dic, entity2idx_dic, rel2idx_dic, args["develop_file"])
    close_data = train_data[:500]

    print(len(train_data))
    print(len(valid_data))

    rel_dic = {}

    for data in train_data:
        triple_targets = data["triples"]
        entity_pool_len = data["entity_pool_size"]
        triple_targets_with_other, _ = make_triple_include_other(entity_pool_len, triple_targets)

        for triple in triple_targets_with_other:
            if triple[1] in rel_dic:
                rel_dic[triple[1]] += 1
            else:
                rel_dic[triple[1]] = 1
    for k in rel_dic:
        print(idx2rel_dic[k], ":", rel_dic[k])

    idx_list = get_batch_idex(train_data)

    save_path = os.path.join("./", args["model_dir"])

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        model = create_model(sess)
        for epoch in range(args["epoch"]):
            for idx in tqdm(range(len(idx_list))):
                start, end = idx_list[idx]
                # print(idx_list[idx])
                #
                # context2idx, context_length, context_entity, context_char2idx, context_chat_length, entity_pool2idx, \
                # entity_pool_length, entity_pool_type, entity_pool_char2idx, entity_pool_charlen, subject, relation, object =\
                # get_batch_data(train_data[start:end])
                #
                # print(np.shape(context2idx))
                # print(np.shape(context_length))
                # print(np.shape(context_entity))
                # print(np.shape(context_char2idx))
                # print(np.shape(context_chat_length))
                # print(np.shape(entity_pool2idx))
                # print(np.shape(entity_pool_length))
                # print(np.shape(entity_pool_type))
                # print(np.shape(entity_pool_char2idx))
                # print(np.shape(entity_pool_charlen))
                # print(np.shape(subject))
                # print(np.shape(relation))
                # print(np.shape(object))

                # sb_loss, re_loss, ob_loss, total_loss = model.train_step(sess, get_batch_data(train_data[start:end]))
                # total_loss = model.train_step(sess, get_batch_data(train_data[start:end]))

                # if idx % 30 == 0:
                #     print(format("\tEpoch %d (%d/%d)" % (epoch + 1, (idx + 1), len(idx_list))))
                #     # print("Total loss: ", total_loss)
                #     print("Subject loss: ", sb_loss, "\nRelarion loss: ", re_loss, "\nObject loss: ", ob_loss,
                #       "\nTotal loss: ", total_loss, )
                object_loss, re_loss, subject_loss, rev_loss, loss = model.train_step(sess, get_batch_data(train_data[start:end]))
                if idx % 30 == 0:
                    print(format("\tEpoch %d (%d/%d)" % (epoch + 1, (idx + 1), len(idx_list))))
                    # print("Total loss: ", total_loss)
                    print("\nObject loss: ", object_loss, "\nRelarion loss: ", re_loss, "\nSubject loss: ",
                          subject_loss, "\nReverse relation loss: ", rev_loss, "\nTotal loss: ", loss)

            if (epoch + 1) % args["save_point"] == 0:
                model.save_model(sess, save_path, epoch+1)

            if (epoch + 1) % 1 == 0:
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
                # target_triple, predict_triple = [], []
                # for idx, _ in enumerate(re_target_list):
                #     re_pred = re_pred_list[idx]
                #     re_target = re_target_list[idx]
                #
                #     sub_target, ob_target, sub_pred, ob_pred = [], [], [], []
                #     for i, _ in enumerate(entity_target_list[idx]):
                #         if i % 2 == 0:
                #             sub_target.append(entity_target_list[idx][i])
                #             sub_pred.append(entity_pred_list[idx][i])
                #         else:
                #             ob_target.append(entity_target_list[idx][i])
                #             ob_pred.append(entity_pred_list[idx][i])
                #     target_triple.append((sub_target, re_target, ob_target))
                #     predict_triple.append((sub_pred, re_pred, ob_pred))
                # f1_measure(target_triple, predict_triple)

            random.shuffle(train_data)
            idx_list = get_batch_idex(train_data)

def test():
    embedding_table, word2idx_dic, idx2word_dic, char2idx_dic, idx2char_dic, entity2idx_dic, idx2entity_dic, \
    rel2idx_dic, idx2rel_dic = read_embedding()
    args["embedding_table"] = embedding_table

    train_data = read_data(word2idx_dic, char2idx_dic, entity2idx_dic, rel2idx_dic, args["train_file"])
    valid_data = read_data(word2idx_dic, char2idx_dic, entity2idx_dic, rel2idx_dic, args["test_file"])
    close_data = train_data[:500]

    idx_list = get_batch_idex(train_data)
    print(len(valid_data))

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        model = create_model(sess)
        print("########## test ############")
        valid_idx_list = get_batch_idex(valid_data)
        object_pred_list, re_pred_list, subject_pred_list, rev_relation_pred_list = [], [], [], []

        for idx in tqdm(range(len(valid_idx_list))):
            start, end = idx_list[idx]
            batch_data = get_batch_data(valid_data[start:end])

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

        micro_performance(valid_data, object_pred_list, re_pred_list, subject_pred_list, rev_relation_pred_list,
                          idx2rel_dic)

def valid_performance(valid_set, object_predicts, relation_predicts, subject_predicts, rev_relation_predicts,
                      idx2rel_dict):
    ignore = [0, 1]
    recall_cnt, precision_cnt, correct_cnt = 0, 0, 0
    other_recall_cnt, other_precision_cnt, other_correct_cnt = 0, 0, 0
    only_other_recall_cnt, only_other_precision_cnt, only_other_correct_cnt = 0, 0, 0
    rel_dic = {}

    # print(len(valid_set))
    # print(len(object_predicts))
    # print(len(relation_predicts))
    # print(len(subject_predicts))
    # print(len(rev_relation_predicts))

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

        triple_targets_with_other, _ = make_triple_include_other(entity_pool_len, triple_targets)
        triple_predicts_with_other, only_other = make_triple_include_other(entity_pool_len, triple_predicts)

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

        for pre_triple in only_other:
            if pre_triple in triple_targets_with_other:
                only_other_correct_cnt += 1

        precision_cnt += len(triple_predicts)
        recall_cnt += len(triple_targets)

        other_precision_cnt += len(triple_predicts_with_other)
        other_recall_cnt += len(triple_targets_with_other)

        only_other_precision_cnt += len(only_other)
        only_other_recall_cnt += len(triple_targets_with_other)

        print("context: ", valid_set[idx]["context"])
        print("targets:", triple_targets)
        triple_index2word(triple_targets, valid_set[idx]["entity_pool"], idx2rel_dict)
        print("predicts:", triple_predicts)
        triple_index2word(triple_predicts, valid_set[idx]["entity_pool"], idx2rel_dict)
        print("==================================================")

    print("dst:", rel_dic)

    if recall_cnt == 0:
        recall = 0
    else:
        recall = correct_cnt/float(recall_cnt)

    if precision_cnt == 0:
        precision = 0
    else:
        precision = correct_cnt/float(precision_cnt)
    if (recall+precision == 0):
        f_1 = 0
    else:
        f_1 = (2*recall*precision)/(recall+precision)

    print("Recall: ", recall)
    print("Precision:", precision)
    print("F1:", f_1)

    if other_recall_cnt == 0:
        recall = 0
    else:
        recall = other_correct_cnt/float(other_recall_cnt)

    if other_precision_cnt == 0:
        precision = 0
    else:
        precision = other_correct_cnt/float(other_precision_cnt)
    if (recall+precision == 0):
        f_1 = 0
    else:
        f_1 = (2*recall*precision)/(recall+precision)

    print("other_Recall: ", recall)
    print("other_Precision:", precision)
    print("other_F1:", f_1)

    if only_other_recall_cnt == 0:
        recall = 0
    else:
        recall = only_other_correct_cnt/float(only_other_recall_cnt)

    if only_other_precision_cnt == 0:
        precision = 0
    else:
        precision = only_other_correct_cnt/float(only_other_precision_cnt)
    if (recall+precision == 0):
        f_1 = 0
    else:
        f_1 = (2*recall*precision)/(recall+precision)

    print("only other_Recall: ", recall)
    print("only other_Precision:", precision)
    print("only other_F1:", f_1)

def triple_index2word(triple_list, entity_pool, idx2rel_dict):
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
        print()

def make_triple_include_other(entity_pool_len, default_triple: list):
    triple_include_other = set()
    only_other = []

    for sub_id in range(entity_pool_len):
        sub_id = sub_id + 2
        only_other.append((sub_id, 1, 1))
        flag = False

        for triple in default_triple:
            if triple[0] == sub_id:
                flag = True
                triple_include_other.add(triple)

        if not flag:
            triple_include_other.add((sub_id, 1, 1))

    # print("default: ", default_triple)
    # print("other: ", triple_include_other)

    return list(triple_include_other), only_other

# def make_triple_include_other(entity_pool_len, default_triple: list):
#     triple_include_other = set()
#     only_other = []
#     print("2")
#
#     for sub_id in range(entity_pool_len):
#         sub_id = sub_id + 2
#         only_other.append((sub_id, 1, 1))
#         flag = False
#
#         for triple in default_triple:
#             if triple[0] == sub_id:
#                 flag = True
#                 triple_include_other.add(triple)
#             elif triple[2] == sub_id:
#                 flag = True
#                 triple_include_other.add(triple)
#
#         if not flag:
#             triple_include_other.add((sub_id, 1, 1))
#
#     # print("default: ", default_triple)
#     # print("other: ", triple_include_other)
#     triple_include_other = list(triple_include_other)
#
#     return triple_include_other, only_other


def f1_measure(targets, predicts):
    # [([s1, s2, s3...], [r1, r2, r3...], [o1, o2, o3...]), (s_li, r_li, o_li)....]
    recall_cnt, precision_cnt, correct_cnt = 0, 0, 0

    def make_triple(tu_list):
        result = []
        end_symbol = 1

        sub_li, rel_li, ob_li = tu_list
        for sub, rel, ob in zip(sub_li, rel_li, ob_li):
            if (sub == end_symbol) or (rel == end_symbol) or (ob == end_symbol):
                break
            result.append((sub, rel, ob))

        return result

    for target, predict in zip(targets, predicts):
        tar_triples = make_triple(target)
        pred_triples = make_triple(predict)
        pred_triples = list(set(pred_triples))

        print("targets:", tar_triples)
        print("predicts:", pred_triples)
        print()

        for pre_triple in pred_triples:
            if pre_triple in tar_triples:
                correct_cnt += 1

        precision_cnt += len(pred_triples)
        recall_cnt += len(tar_triples)

    if recall_cnt == 0:
        recall = 0
    else:
        recall = correct_cnt/float(recall_cnt)

    if precision_cnt == 0:
        precision = 0
    else:
        precision = correct_cnt/float(precision_cnt)
    if (recall+precision == 0):
        f_1 = 0
    else:
        f_1 = (2*recall*precision)/(recall+precision)

    print("Recall: ", recall)
    print("Precision:", precision)
    print("F1:", f_1)

def micro_performance(valid_set, object_predicts, relation_predicts, subject_predicts, rev_relation_predicts,
                      idx2rel_dict):
    ignore = [0, 1]
    recall_cnt, precision_cnt, correct_cnt = 0, 0, 0
    other_recall_cnt, other_precision_cnt, other_correct_cnt = 0, 0, 0
    only_other_recall_cnt, only_other_precision_cnt, only_other_correct_cnt = 0, 0, 0
    rel_dic = {}

    # print(len(valid_set))
    # print(len(object_predicts))
    # print(len(relation_predicts))
    # print(len(subject_predicts))
    # print(len(rev_relation_predicts))
    micro_re, micro_pr, micro_cr = {}, {}, {}

    for idx, (objects, relations, subjects, rev_relations) in enumerate(zip(object_predicts, relation_predicts,
                                                                            subject_predicts, rev_relation_predicts)):
        triple_targets = valid_set[idx]["triples"]
        triple_predicts = set()
        entity_pool_len = valid_set[idx]["entity_pool_size"]

        # if entity_pool_len != 2:
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

        triple_targets_with_other, _ = make_triple_include_other(entity_pool_len, triple_targets)
        triple_predicts_with_other, only_other = make_triple_include_other(entity_pool_len, triple_predicts)

        for triple in triple_targets_with_other:
            if triple[1] in rel_dic:
                rel_dic[triple[1]] += 1
            else:
                rel_dic[triple[1]] = 1

        for pre_triple in triple_predicts:
            if pre_triple in triple_targets:
                correct_cnt += 1

        for pre_triple in only_other:
            if pre_triple in triple_targets_with_other:
                only_other_correct_cnt += 1

        for pre_triple in triple_predicts_with_other:
            relation_idx = pre_triple[1]
            if pre_triple in triple_targets_with_other:
                other_correct_cnt += 1

                ## micro corrects
                if relation_idx in micro_cr:
                    micro_cr[relation_idx] += 1
                else:
                    micro_cr[relation_idx] = 1
            ## micro precisions
            if relation_idx in micro_pr:
                micro_pr[relation_idx] += 1
            else:
                micro_pr[relation_idx] = 1

        precision_cnt += len(triple_predicts)
        recall_cnt += len(triple_targets)

        other_precision_cnt += len(triple_predicts_with_other)
        other_recall_cnt += len(triple_targets_with_other)

        only_other_precision_cnt += len(only_other)
        only_other_recall_cnt += len(triple_targets_with_other)

        print("context: ", valid_set[idx]["context"])
        print("targets:", triple_targets)
        triple_index2word(triple_targets, valid_set[idx]["entity_pool"], idx2rel_dict)
        print("predicts:", triple_predicts)
        triple_index2word(triple_predicts, valid_set[idx]["entity_pool"], idx2rel_dict)
        print("==================================================")

    print(rel_dic)
    print(micro_pr)
    print(micro_cr)
    print(other_recall_cnt)
    print(other_precision_cnt)
    print(other_correct_cnt)


    for k in rel_dic:
        relation = idx2rel_dict[k]
        mre = micro_cr[k]/float(rel_dic[k])
        mpc = micro_cr[k]/float(micro_pr[k])
        mf1 = (2*mre*mpc)/(mre+mpc)
        print(relation, "micro f1:", mf1)
        print("===============================")


    if recall_cnt == 0:
        recall = 0
    else:
        recall = correct_cnt/float(recall_cnt)

    if precision_cnt == 0:
        precision = 0
    else:
        precision = correct_cnt/float(precision_cnt)
    if (recall+precision == 0):
        f_1 = 0
    else:
        f_1 = (2*recall*precision)/(recall+precision)

    print("Recall: ", recall)
    print("Precision:", precision)
    print("F1:", f_1)

    if other_recall_cnt == 0:
        recall = 0
    else:
        recall = other_correct_cnt/float(other_recall_cnt)

    if other_precision_cnt == 0:
        precision = 0
    else:
        precision = other_correct_cnt/float(other_precision_cnt)
    if (recall+precision == 0):
        f_1 = 0
    else:
        f_1 = (2*recall*precision)/(recall+precision)

    print("other_Recall: ", recall)
    print("other_Precision:", precision)
    print("other_F1:", f_1)

    if only_other_recall_cnt == 0:
        recall = 0
    else:
        recall = only_other_correct_cnt/float(only_other_recall_cnt)

    if only_other_precision_cnt == 0:
        precision = 0
    else:
        precision = only_other_correct_cnt/float(only_other_precision_cnt)
    if (recall+precision == 0):
        f_1 = 0
    else:
        f_1 = (2*recall*precision)/(recall+precision)

    print("only other_Recall: ", recall)
    print("only other_Precision:", precision)
    print("only other_F1:", f_1)

if __name__=="__main__":
    if args["is_train"]:
        print("train model...")
        train()
    else:
        print("test...")
        test()
    # embedding_table, word2idx_dic, idx2word_dic, char2idx_dic, idx2char_dic, entity2idx_dic, idx2entity_dic, \
    # rel2idx_dic, idx2rel_dic = read_embedding()
    # all_data = read_data(word2idx_dic, char2idx_dic, entity2idx_dic, rel2idx_dic)
    #
    # context2idx, context_length, context_entity, context_char2idx, context_chat_length, entity_pool2idx, \
    # entity_pool_length, entity_pool_type, entity_pool_char2idx, entity_pool_charlen, subject, relation, object =\
    # get_batch_data(all_data)
    #
    # print(np.shape(context2idx))
    # print(np.shape(context_length))
    # print(np.shape(context_entity))
    # print(np.shape(context_char2idx))
    # print(np.shape(context_chat_length))
    # print(np.shape(entity_pool2idx))
    # print(np.shape(entity_pool_length))
    # print(np.shape(entity_pool_type))
    # print(np.shape(entity_pool_char2idx))
    # print(np.shape(entity_pool_charlen))
    # print(np.shape(subject))
    # print(np.shape(relation))
    # print(np.shape(object))
