# Relation Extraction Submission

## Summary

- 이전에 연구했던 논문 ([Deep Context- and Relation-aware Learning for Aspect-based Sentiment Analysis](https://aclanthology.org/2021.acl-short.63/)) 의 핵심 아이디어 중 하나인 Context-aware Multi-task Learning 방식을 Relation Extraction 태스크에 맞게 디자인하고 실험하였습니다.
- 기존 Relation Extraction 관련 논문 ([Matching the Blanks: Distributional Similarity for Relation Learning](https://aclanthology.org/P19-1279/)) 의 approach 중 일부(Entity Markers)를 활용했습니다.
- 최종 제출은 Ensemble(Soft Voting) 방식을 적용하였습니다.

## Approach
### 1. Context-aware Multi-task Learning
- Inspired by [Deep Context- and Relation-aware Learning for Aspect-based Sentiment Analysis](https://aclanthology.org/2021.acl-short.63/) (Oh et al., ACL-IJCNLP 2021)
- 핵심 아이디어
    - entity 단어가 직접 주어지지 않더라도 문맥을 통해 relation을 추정할 수 있게 하여 contextual information을 직접적으로 고려하게 하는 것이 목적입니다.
    - 아래와 같은 보조 태스크들을 추가하여 Multi-task Learning 방식으로 모델을 학습하였습니다.
        - subject를 masking한 문장을 모델의 입력으로 넣어 relation을 예측
        
            ```
            -- Data --
            Subject: "강경준"
            Object: "장신영"
            Sentence: 배우 장신영 씨와 남편 강경준 씨가 …
            
            --Input & Target--
            Input: "[SBJ] [SEP] 장신영 [SEP] 배우 장신영 씨와 남편 [SBJ] 씨가 …"       
            Target: "per:spouse"
            ```  
        - object를 masking한 문장을 모델의 입력으로 넣어 relation을 예측
            ```
            -- Data --
            Subject: "강경준"
            Object: "장신영"
            Sentence: 배우 장신영 씨와 남편 강경준 씨가 …
            
            --Input & Target--
            Input: "강경준 [SEP] [OBJ] [SEP] 배우 [OBJ] 씨와 남편 강경준 씨가 …"       
            Target: "per:spouse"
            ```  
    - 각 sub-task들과 main-task는 Bert와 Classifier를 공유합니다.

- 참고한 논문의 PRD(Pairwise Relation Discrimination) 방식도 실험해 보면 좋을 것 같습니다.
    - ex) aspect-opinion pair 여부를 discrimination하는 태스크를 subject-object pair 여부를 discrimination하는 태스크로 바꿔서 적용 가능
    - 참고 논문의 핵심 아이디어는 Context-aware Multi-task Learning과 Relation-aware Multi-task Learning 입니다.
    - 위 approach는 전자의 방식을 반영했다고 볼 수 있는데 후자의 방식도 효과가 있는지 확인해 볼만할 것 같습니다.


### 2. Entity Markers 
- Inspired by [Matching the Blanks: Distributional Similarity for Relation Learning](https://aclanthology.org/P19-1279/) (Soares et al., ACL 2019)
- 모델의 입력을 아래 예제와 같은 형태로 변형하였습니다.

    ```
    -- Data --
    subject: 강경준
    object: 장신영
    
    --Input Format--
    Input: "배우 [OBJ]장신영[/OBJ] 씨와 남편 [SBJ]강경준[/SBJ] 씨가 …"       
    ```
- Context-aware Multi-task Learning 방식의 sub-task에 대해 실험할 때는 아래와 같이 디자인했습니다.
    ```
    Subject를 Mask하는 sub-task의 경우
    -- Data --
    subject: 강경준
    object: 장신영
    
    --Input Format--
    Input: "배우 [OBJ]장신영[/OBJ] 씨와 남편 [SBJ][ENT][/SBJ] 씨가 …"       
    ```

### 3. Ensemble based on Soft Voting
- inference 결과 파일의 probs를 이용하여 Soft Voting 기반의 Ensemble 기법을 적용하였습니다.

## Experimental results
### 1. base LM 선정방법 및 Model Architecture
- 제공된 베이스라인 코드를 이용하여 실험하였습니다.
    - kleu-bert
    - koelectra-v3-discriminator
    - tunib-electra
- kleu-bert의 성능이 가장 좋아서 kleu-bert를 LM으로 사용하였습니다.
- huggingface의 BertForSequenceClassification 모델을 사용하였습니다.
- optimizer로 [AdamP](https://github.com/clovaai/AdamP)를 사용하였습니다.

### 2. Best Results on Test Dataset
- 학습 과정에서 train data 전체를 사용했습니다.
- Context-aware Multi-task Learning 방법과 Entity Markers 방법을 모두 적용한 모델과 baseline 모델을 앙상블하여 제출하였습니다.
    - test_submission_1: 각각 5 run 사용, 총 10개에 파일에 대해 Soft Voting Ensemble
    - test_submission_2: 모두 적용한 모델 10 run 사용, baseline 모델 5 run 사용, 총 15개 파일에 대해 Soft Voting Ensemble

- 결과

    |                   |    **micro f1**    |       **auprc**        |
    | :---------------- | :----------------: | :--------------------: |
    | test_submission_1 |     **68.9590**    |         75.9100        |
    | test_submission_2 |       68.8270      |       **76.0670**      |
    


### 3. Ablation study
- train:valid = 7:3 랜덤 분할, 5 run average
- 실험 결과

    |                       |    **micro f1**    |       **auprc**        |      **acc**       |
    | :-------------------- | :----------------: | :--------------------: | :----------------: |
    | baseline              |       82.02        |         74.81          |       80.20        |
    | + C-MTL               |     ***83.22**     |         ***76.89**     |       ***81.22**   |
    | + MTB-EM              |       *82.57       |         75.64          |       *80.74       |
    | + C-MTL + MTB-EM      |       *83.03       |         *76.73         |       *81.13       |
    - "*" denotes statistical significance between the baseline and itself (p-value<0.05).
    - C-MTL: Context-aware Multi-task Learning 제안 방법을 적용한 모델
    - MTB-EM: Entity Markers 제안 방법을 적용한 모델
    - baseline: 현재 코드 베이스에서 MTB-EM과 C-MTL 방식을 적용하지 않은 모델

### 4. Analysis
- C-MTL 방식 단독으로 사용했을 때보다 C-MTL과 MTB-EM 방식을 같이 사용했을 때 성능이 더 저조한 이유?
    - MTB-EM 방식은 special token으로 entity를 감싸는 방식인데 main-task에서는 두 special token 사이에 의미적으로 중요한 단어가 있다고 학습하게 될 가능성이 큽니다.
    - 하지만 sub-task에서는 두 special token 사이에 의미 정보를 가진 token이 없습니다. (special token으로 치환했기 때문)
    - 정확한 이유는 검증을 해 봐야겠지만 이 부분에서 혼란이 일어나는 것이 아닌가 추측됩니다.
        - C-MTL만 사용할 때는 두 special token 사이에 중요한 단어가 있다고 학습하게 되는 것이 아니라 문장 내의 entity 단어가 special token 자체로 바뀌기 때문에 위와 같은 문제는 없을 듯 합니다.


## Dropped Approach
### 1. Curriculum Learning based on Hard Voting
- Flow
    1. 학습 데이터를 5분할 하여 각각 5개 모델 학습
    2. 각 모델의 학습에 사용하지 않은 train 데이터들을 inference
    3. 각 example에 대해 정답을 맞힌 모델 수를 기반으로 내림차순 정렬 (Hard Voting)
    4. 정렬된 데이터를 이용해 shuffle을 사용하지 않고 모델 학습
    
- 성능이 좋지 않아서 drop 하였습니다.
- 이유에 대해 따로 검증하지는 않았지만 hard-voting 방식을 사용해서 난이도가 계단식으로 상승하게 되므로 training_loss도 상승 구간에서 급격히 증가하는 현상이 나타났습니다. 이 부분이 오히려 모델에 혼란을 야기한 것 같습니다.
- 향후 soft-voting 방식으로 다시 시도해 볼 수 있을 듯 합니다.

## Instructions
### File & Code Structure
```
upstage_re_submission
└───config                      # 학습에 필요한 파라미터들 설정
│   │   hparams.py
│
└───data                        # 데이터 셋 (7:3 랜덤 분할)
│   └───train
│   │   │   data.csv
│   └───dev
│   │   │   data.csv
│   └───test
│   │   │   data.csv
│   │   dataset.py              # Dataset 클래스를 상속 / get_item 함수에서 전처리
│
└───models
│   │   model.py                # 모델 정의
│   
└───predictions                 # 결과 출력 디렉토리
│   │   test_submission_1.csv   # 최종 제출물
│   │   test_submission_2.csv   # 최종 제출물
│   
└───utils                   
│   │   checkpointing.py        # 모델 체크포인트 저장
│   
│   dict_label_to_num.pkl
│   dict_num_to_label.pkl
│   ensemble.py                 # inference 결과물을 이용해 soft-voting ensemble을 적용
│   inference.py                # 제공받은 inference.py를 본 코드 베이스에 맞게 수정
│   load_data.py                # 제공받은 파일 / inference 과정에서 test_data를 load하는 용도로만 사용
│   main.py                     # 실행 시작 지점 / argument 정의 등
│   predictor.py                # evaluation 함수 내재
│   README.md               
│   requirements.txt
│   train.py                    # trainer

```

### Environment Settings
- OS: Ubuntu 18.04
- GPU: 1 * V100 16GB
- CUDA 11.1
- python 3.8
```
pip install -r requirements.txt
```
### How To Run
```
python main.py
```

## Acknowledgments
- [Deep Context- and Relation-aware Learning for Aspect-based Sentiment Analysis](https://aclanthology.org/2021.acl-short.63/)
- [Matching the Blanks: Distributional Similarity for Relation Learning](https://aclanthology.org/P19-1279/)
- [Transformers - huggingface](https://huggingface.co/transformers/)
- [Pytorch](https://pytorch.org/)