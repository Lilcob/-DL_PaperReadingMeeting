# Deep learning Paper Reading Meeting Archive
This page is an archive of the Deep Learning Paper Reading Meeting.
you would like to attend the meeting or have any questions,
Write in the GitHub issue table or email us at 'tfkeras@kakao.com'
 
| Tasks | Paper | Link| Performance Index| Highlights| 
|:---------------|:-------------:|:-------------:|-------------:|-------------:|
| NLP | Attention is all you need |[Youtube](https://www.youtube.com/watch?v=EyXehqvkfF0)<br> [Paper](https://arxiv.org/pdf/1706.03762.pdf)|Abstractive Text Summarization, Constituency Parsing|완전한 어텐션 기반(Transformer) 모델을 제안하여, RNN/CNN을 사용하지 않고도 SOTA 달성.|
| NLP | BERT | [Youtube](https://www.youtube.com/watch?v=vo3cyr_8eDQ&t=7s) <br> [paper](https://arxiv.org/pdf/1810.04805.pdf)| Autoencoding Transformers, Transformers, Language Models|마스크드 LM과 NSP 기법을 통해 범용적인 언어 이해 성능을 혁신적으로 향상.|
| NLP | ERNIE| [Youtube](https://www.youtube.com/watch?v=_K6izzEeKYg&t=985s) <br> [paper](https://arxiv.org/pdf/1905.07129.pdf)| Chinese Named Entity Recognition, Chinese Sentence Pair Classification |지식 그래프를 통합하여 중국어에 특화된 언어 모델 성능을 높임.|
| NLP | RoBERTa | [Youtube](https://www.youtube.com/watch?v=GiotNYiTiMw&t=6s) <br> [paper](https://arxiv.org/pdf/1907.11692.pdf)| Autoencoding Transformers, Transformers|더 많은 데이터와 긴 학습, 대규모 배치를 활용해 BERT보다 높은 성능 달성.|
| NLP | XLNET | [Youtube](https://www.youtube.com/watch?v=29oxqDRaPNo&t=898s) <br> [paper](https://arxiv.org/pdf/1906.08237.pdf)| Autoregressive Transformers, Transformers|순열 기반 언어 모델링으로 BERT의 NSP 한계를 극복하고 성능을 개선.|
| NLP | SentenceBert |[Youtube](https://www.youtube.com/watch?v=izCeQOOuZpY&t=291s)| Linear-Probe Classification, Semantic Similarity |BERT를 문장 임베딩에 최적화하여 문장 간 유사도 계산을 간단하면서도 효과적으로 수행.|
| NLP | Defending Against neural fake news | [Youtube](https://www.youtube.com/watch?v=cjjfJhYqyeg&t=10s) | Fake News Detection, Text Generation |GPT류 모델이 생성한 텍스트와 실제 뉴스 텍스트를 구분하기 위한 방법론 제안 및 실험.|
| NLP | Transformer-XL |[Youtube](https://www.youtube.com/watch?v=2SDI7hUoDSU&t=3s) <br> [blog](https://ghk829.github.io/whitecross.github.com//transformer-xl) |Language Modelling |	긴 시퀀스 처리를 위해 세그먼트 메모리를 순환적으로 연결하는 새로운 구조 제안.|
| NLP | Understanding back translation at scale | [Youtube](https://www.youtube.com/watch?v=htzBkroOLg4&t=12s) <br> [blog]( https://dev-sngwn.github.io/2020-01-07-back-translation/ ) | Machine Translation, Translation|대규모 역번역(back-translation) 실험을 통해 번역 품질 향상을 체계적으로 분석.|
| NLP | Deep Contextualized Word Representations |[Youtube](https://www.youtube.com/watch?v=Vc13QVAKyGk)|Citation Intent Classification , Conversational Response Selection|문맥에 따라 단어 임베딩이 동적으로 변하는 ELMo를 통해 다양한 NLP 태스크 성능 향상.|
| NLP | Univiersal LM Fine-tuning for text classification | [Youtube](https://www.youtube.com/watch?v=ZJKtwX2LSbY&t=1173s)|Classification, General Classification|범용 언어 모델 사전학습 후 파인튜닝 방식을 제안해 소규모 데이터에서도 뛰어난 성능 달성.|
| NLP | Subword-level Word Vector Representations for Korean | [Youtube](https://www.youtube.com/watch?v=QR5XFn5rdMQ)| Document Classification, Language Modelling|한국어 형태론에 맞춰 서브워드 단위 임베딩을 제안하여 분절 문제를 효과적으로 해결.|
| NLP | A Decomposable Attention Model for Natural Language Inference | [Youtube](https://youtu.be/8FcJtvCxI68)| Natural Language Inference|간단한 어텐션 구조만으로도 NLI 태스크에서 준수한 성능을 달성한 모델 제안.|
| NLP | Reformer | [Youtube](https://youtu.be/B6eLtGKgK68)|Transformers|LSH 기반 어텐션으로 긴 시퀀스 처리 시 메모리와 연산량을 크게 절감.|
| NLP | Neural Machine Translation by Jointly Learning to Align and Translate | [Youtube](https://youtu.be/l9pWT6BHpj0)|Dialogue Generation, Machine Translation|기념비적인 연구로, NMT에서의 어텐션(align) 메커니즘을 처음 제안.|
| NLP | ELECTRA | [Youtube](https://youtu.be/BGRculoppT8)| Autoencoding Transformers, Transformers|제너레이터-디스크리미네이터 구조로 적은 연산으로 효율적인 사전학습 구현.|
| NLP | SBERT-WK | [Youtube](https://youtu.be/qXZ80xn8DDU)| Semantic Textual Similarity, Sentence Embedding|BERT 내부 레이어 가중치(Weighted-KMeans) 조합으로 문장 임베딩 생성.|
| NLP | Revealing the Dark Secrets of BERT | [Youtube](https://youtu.be/gcar30nhgqQ)|BERT|BERT 내부 표현 구조와 잠재적 취약점을 심층 분석한 연구.|
| NLP | PEGASUS | [Youtube](https://youtu.be/JhGmeQBbDdA)|Abstractive Text Summarization|문장 누락(gap-sentence) 기반 사전학습 기법으로 요약 태스크에 특화된 성능 제시.|
| NLP | Document-level Neural Machine Translation with Inter-Sentence Attention | [Youtube](https://youtu.be/4QTydWc2xYs) |Machine Translation, Translation|문서 단위 전체 문맥(문단 간 관계)을 고려해 번역 품질을 높임.|
| NLP |  Phrase-Based & Neural Unsupervised Machine | [Youtube](https://www.youtube.com/watch?v=G3rUzTFf_l4) |Translation, Unsupervised Machine Translation|무감독 기법으로 평행 데이터 없이 양방향 번역을 학습하는 모델 제안.|
| NLP | BART | [Youtube](https://youtu.be/VmYMnpDLPEo) |Sequence To Sequence Models, Transformers|디노이징 오토인코더 방식의 사전학습으로 요약·생성 분야에서 강력한 성능 발휘.|
| NLP | BAE | [Youtube](https://youtu.be/ukkcBtvPB3k) | Adversarial Attack, Adversarial Text|토큰 치환 방식으로 텍스트를 교란해 분류 모델을 공격·취약점 연구.|
| NLP | A Generative Model for Joint Natural Language Understanding and Generation | [Youtube](https://youtu.be/Om0aAcZfjxE)| Natural Language Understanding, Task-Oriented Dialogue Systems|언어 이해(NLU)와 생성(NLG)을 단일 모델에서 통합적으로 수행하는 대화 시스템 제안.|
| NLP | Learning Contextual Representations for Semantic Parsing with Generation-Augmented Pre-Training | [Youtube](https://youtu.be/L_5plaUpBaA)| Language Modelling, self-Supervised Learning|생성 기반 사전학습을 활용하여 의미파싱(semantic parsing) 정확도를 향상.|
| NLP | Graph Attention Networks | [Youtube](https://youtu.be/shdNuppfClU)| Document Classification, Graph Attention|그래프에서 어텐션 기법을 적용해 노드·문서 분류 성능을 개선한 모델.|
| NLP | Switch Transformers | [Youtube](https://youtu.be/82kpTjm-M_g)| Language Modelling, Question Answering|초대규모 파라미터 모델에 전문가 혼합(MoE)을 적용해 효율적 스케일링 구현.|
| NLP | DeText: A Deep Text Ranking Framework with BERT | [Youtube](https://youtu.be/tE_1uiaUf1k)|Ranking|검색 및 추천 시스템에서 BERT 스타일 랭킹 모델을 적용해 정확도 향상.|
| NLP | Face book Chat bot , Blender bot | [Youtube](https://youtu.be/wd64FDWCmDs)|Chatbot|지식, 인격 등 여러 스킬을 혼합한 오픈도메인 챗봇으로 대화 맥락을 풍부하게 유지.|
| NLP | Extracting Training Data from Large Language Models | [Youtube](https://youtu.be/NGoDUEz3tZg)| Language Modelling|대형 언어 모델이 학습 데이터를 “기억” 및 유출할 수 있는 프라이버시 이슈 연구.|
| NLP | Longformer: The Long-Document Transformer | [Youtube](https://youtu.be/i7aiBMDExmA)| Language Modelling, Question Answering|희소 어텐션(sparse attention)을 적용해 긴 문서를 효율적으로 처리.|
| NLP | Visualizing and Measuring the Geometry of BERT | [Youtube](https://youtu.be/4DXU3MaGHcU)|Word Embeddings|BERT 임베딩 공간의 기하학적 특성과 분포를 시각화 및 정량 분석.|
| NLP | Encode, Tag, Realize High Precision Text Editing | [Youtube](https://youtu.be/bs_GjHGV5T4)| Abstractive Text Summarization, Sentence Fusion|단계별(인코드→태깅→실현) 접근으로 정밀한 텍스트 편집 및 요약 방안 제안.|
| NLP | multimodal transformer for unaligned multimodal language sequences | [Youtube](https://youtu.be/uEwxvQ9lXAQ)| Multimodal Sentiment Analysis, Time Series|서로 정렬되지 않은 멀티모달 입력(음성·영상 등) 처리를 Transformer로 수행하여 감정분석·시계열 분석.|
| NLP | SCGPT : Few-shot Natural Language Generation for Task-Oriented Dialog | [Youtube](https://youtu.be/BAZxrp2nrF8)| Data-to-Text Generation, Few-Shot Learning|대화 시스템에서 소량의 데이터로 텍스트 생성(task-oriented)을 수행하는 GPT 기반 프레임워크.|
| NLP | ColBERT: Efficient and Effective Passage Search viaContextualized Late Interaction over BERT | [Youtube](https://youtu.be/5mynfZA2t7U)|Document Ranking, Information Retrieval|BERT 임베딩을 기반으로 후반부(Late) 상호작용을 적용해 효율적인 패시지 검색을 구현.|
| NLP | Restoring and Mining the Records of the Joseon Dynasty via Neural LanguageModeling and Machine Translation | [Youtube](https://youtu.be/BkyVMuvO5bE)|Language Modelling, Machine Translation|조선왕조실록 등 옛 문서를 복원하고 번역하기 위해 NLM·MT 기술을 활용.|
| NLP | Improving Factual Completeness and Consistency of Image to Text Radiology Report Generation | [Youtube](https://youtu.be/OZLcZML7SO8)|Natural Language Inference, Text Generation|영상→텍스트 변환 시 사실성(팩추얼)과 일관성을 높여 의료 보고서 품질 향상.|
| NLP | FinBERT | [Youtube](https://youtu.be/FQN8sOi1PTI)|Language Modelling, Pretrained Language Models|금융 도메인(감성 분석 등)에 특화된 BERT 변형 모델로 높은 성능 달성.|
| NLP | LayoutLM: Pre-training of Text and Layout for Document Image Understanding | [Youtube](https://youtu.be/3HqAgrcPprQ)|Document Image Classification, Document Layout Analysis|문서 이미지에서 텍스트와 레이아웃 정보를 동시에 학습해 인식 정확도 향상.|
| NLP | Query Suggestions as Summarization in Exploratory Search | [Youtube](https://youtu.be/AVTZq2rWS0k)|Query suggestions|	쿼리 제안을 요약(summarization) 방식으로 접근해 탐색적 검색을 지원.|
| NLP | H-Transformer-1D Paper: Fast One Dimensional Hierarchical Attention For Sequences | [Youtube](https://youtu.be/P19XuHOVyVk)|Language Modelling|계층적(hierarchical) 어텐션을 적용해 1D 시퀀스 처리 속도를 높인 모델.|
| NLP | End-to-End Progressive Multi-Task Learning Framework for Medical Named Entity Recognition and Normalization | [Youtube](https://youtu.be/kmuET_G1brY)| Knowledge Graphs, Medical Named Entity Recognition|의료 개체 인식과 정규화를 점진적 멀티태스크 학습으로 통합 수행.|
| NLP | DISEASES : Text mining and data integration of disease–gene associations | [Youtube](https://youtu.be/_2rfSxBSFnc)|Text mining, Data integration, Information extraction|생물의학 텍스트를 활용해 질병–유전자 연관성을 통합적으로 발굴·구현.|
| NLP | RoFormer: Enhanced Transformer with Rotary Position Embedding | [Youtube](https://youtu.be/tRe2XHF6UbQ)|Semantic Text Matching|Rotary Position Embedding 기법으로 위치 인코딩을 개선해 매칭 성능 향상.|
| NLP | A Multiscale Visualization of Attention in the Transformer Model | [Youtube](https://youtu.be/Gl2WUBQYEfg)|Transformer|	멀티스케일 어텐션 시각화로 Transformer 동작과 메커니즘 파악에 도움.|
| NLP | CAST: Enhancing Code Summarization with Hierarchical Splitting and Reconstruction of Abstract Syntax Trees | [Youtube](https://youtu.be/h8YBJzuBSsA)|Software Engineering|코드 요약 시 AST를 계층적으로 분할·재구성하여 더 간결하고 정확한 요약 제공.|
| NLP | MERL: Multimodal Event Representation Learning in Heterogeneous Embedding Spaces | [Youtube](https://youtu.be/shnfzksjm1M)|Language Grounding, Multi-modal NLP|서로 다른 모달(텍스트, 이미지 등)을 동일 임베딩 공간에서 이벤트로 학습.|
| NLP | Big Bird - Transformers for Longer Sequences | [Youtube](https://youtu.be/vV7fN1eUqbI)| Linguistic Acceptability, Natural Language Inference|희소 어텐션으로 4k 토큰 이상의 긴 입력을 처리 가능하게 만든 모델.|
| NLP | Decoding-Enhanced BERT with Disentangled Attention | [Youtube](https://youtu.be/hNTkpNk7v-I)| Common Sense Reasoning, Coreference Resolution|디코딩 모듈과 분리 어텐션을 결합해 상식 추론과 코리퍼런스 해결 성능을 높임.|
| NLP | SentiPrompt: Sentiment Knowledge Enhanced Prompt-Tuning for Aspect-Based Sentiment Analysis | [Youtube](https://youtu.be/V9DlySYag_Q)|Aspect-Based Sentiment Analysis, Language Modelling|감성 지식을 프롬프트 튜닝에 접목해 ABSA(Aspect 기반 감성분석) 성능 향상.|
| NLP | IMPROVING BERT FINE-TUNING VIA SELF-ENSEMBLE AND SELF-DISTILLATION | [Youtube](https://youtu.be/3PVada1CVYQ)|Natural Language Inference, Text Classification|여러 BERT 체크포인트를 앙상블 후 셀프 디스틸로 성능 및 안정성 개선.|
| NLP | ACHIEVING HUMAN PARITY ON VISUAL QUESTION ANSWERING | [Youtube](https://youtu.be/Gcbf0M0Qx9U)| question-answering,Question Answering|시각적 질의응답(VQA) 분야에서 인간 수준에 근접하거나 달성한 모델.|
| NLP | Deep Encoder, Shallow Decoder: Reevaluating non- autoregressive machine translation | [Youtube](https://youtu.be/sP7Ue_MbiKE)|Knowledge Distillation, Machine Translation|인코더를 깊게, 디코더를 얕게 설계하여 비자동회귀 MT(non-AR MT) 성능 향상 제시.|
| NLP | LaMDA : Language Models for Dialog Applications | [Youtube](https://youtu.be/re2AiBnFGGk)| Information Retrieval|다중 턴 대화를 위한 대규모 언어모델, 풍부한 대화 능력(오픈엔드).|
| NLP | Competition-Level Code Generation with AlphaCode | [Youtube](https://youtu.be/c7jCWU_ChjE)|Code Generation|프로그래밍 경진대회 수준의 코드 솔루션을 생성하는 모델.|
| NLP | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | [Youtube](https://youtu.be/IUwfhbZYTno)|Knowledge-Intensive NLP|외부 지식을 검색(Retrieval) 후 생성(Generation)에 활용해 복잡한 질의 해결.|
| NLP | SimCLS: A Simple Framework for Contrastive Learning of Abstractive Summarization | [Youtube](https://youtu.be/3CPfP7oaXO8)|Abstractive Summarization|요약 결과를 대조학습으로 평가·개선해 더 높은 품질의 추상적 요약 가능.|
| NLP | Graph-Bert: Only Attention is Needed for Learning Graph Representations | [Youtube](https://youtu.be/2kQDL-byr0k)|Graph Representations|그래프 구조에 직접 어텐션만으로 임베딩을 학습, GNN 없이도 성능 확보.|
| NLP | SimCSE:Simple Contrastive Learning of Sentence Embedding | [Youtube](https://youtu.be/RToEmUYyGO4)|Sentence Embedding|최소한의 데이터 증강과 대조학습만으로 문장 임베딩 품질을 대폭 향상.|
| NLP | Typical decoding for Natural Language | [Youtube](https://youtu.be/1_xw30L31n8)|Text Generation|생성 과정에서 “typical decoding”을 도입해 다양성과 품질의 균형을 꾀함.|
| NLP | Perceiver IO : A GENERAL ARCHITECTURE FOR STRUCTURED INPUTS & OUTPUTS | [Youtube](https://youtu.be/rRQ5uBWB9EQ)|Multimodal Processing|다양한 형태의 입·출력을 일관된 Transformer 구조로 처리할 수 있는 범용 프레임워크.|
| NLP | Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference | [Youtube](https://youtu.be/tdB0Y6hMtcg)|Text Classification, Natural Language Inference| 클로즈(cloze) 스타일의 질문을 이용해 소량의 데이터로 분류·추론 태스크를 수행|
| NLP | Learning to Generalize to More:Continuous Semantic Augmentation for Neural Machine Translation | [Youtube](https://youtu.be/zy3jh-ygXFk)|Machine Translation|연속적 의미 증강(continuous semantic augmentation)을 통해 번역 모델의 일반화 성능 제고.|
| NLP | Multitask Prompted Training Enables Zero-Shot Task | [Youtube](https://youtu.be/AttJCATMhRo)|Zero-Shot Learning|멀티태스크 프롬프트 학습으로, 새로운 태스크에 대한 제로샷 성능을 높임.|
| NLP | Efficient Passage Retrieval with Hashing for Open-domain Question Answering(BPR) | [Youtube](https://youtu.be/ESzmoZU-QfQ)|Open-domain QA|해싱을 활용한 효율적 패시지 검색으로 오픈도메인 QA 시스템 성능 향상.|
| NLP | LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS | [Youtube](https://youtu.be/BJqwmDpa0wM)|Language Modelling, LLM finetuning|대형 언어 모델에 저순위 행렬(LoRA)을 추가해 파인튜닝 비용을 절감하는 기법.|
| NLP | Weakly-supervised Text Classification Based on Keyword Graph | [Youtube](https://youtu.be/JYlRXo3a2R8)|Text Classification|키워드 그래프를 이용한 약지도(weakly-supervised) 텍스트 분류 기법.|
| NLP | Improving the Quality Trade-Off for Neural Machine Translation Multi-Domain Adaptation | [Youtube](https://youtu.be/c8-aLsXqvus)|Machine Translation|다중 도메인 번역 적응 시 품질과 비용 간 균형을 개선하는 접근 제안.|
| NLP | AEDA: An Easier Data Augmentation Technique for Text Classification | [Youtube](https://youtu.be/E9rcpKtGars)| Text Classification| 문장부호(punctuation) 삽입만으로 간단히 데이터 증강을 수행하는 기법 제안.|
| NLP | GTRANS-Grouping and Fusing Transformer Layers for Neural Machine Translation | [Youtube](https://youtu.be/DgqnbETZNZw) | Machine Translation|Transformer 레이어를 그룹화·융합해 번역 성능 및 효율을 동시에 추구.|
| NLP | SPLADE : Sparse Lexical and Expansion Model for First Stage Ranking | [Youtube](https://youtu.be/OvVajh6yPEg)| Information Retrieval|문서 랭킹의 첫 단계에서 희소 레키컬 확장을 활용해 성능을 높인 모델.|
| NLP | Debiased Contrastive learning of Unsupervised Sentence Representation | [Youtube](https://youtu.be/SJUZFEE5ELw)| Sentence Representation|비지도 문장 임베딩에서 대조학습을 통해 편향(spurious correlation)을 완화.|
| NLP | Calibrating Sequence Likelihood Improves Conditional Language Generation | [Youtube](https://youtu.be/yLifwXr3Q9k)| Conditional Language Generation|시퀀스 확률 보정을 통해 조건부 텍스트 생성 결과의 일관성과 품질 개선.|
| NLP | CHAIN-OF-THOUGHT PROMPTING ELICITS REASONING IN LARGE LANGUAGE MODELS | [Youtube](https://youtu.be/9JBG2hAkP2E)|	Reasoning|	단계별 추론(Chain-of-Thought)을 유도해 대형 언어모델의 논리적 사고력 강화.|
| NLP | A Quantitative Analysis of Statistical and Graph-Based Term Weighting Schemes for Keyword Extraction | [Youtube](https://youtu.be/YHxoNhCWtic)| Keyword Extraction|통계·그래프 기반 가중치 스킴을 비교하여 키워드 추출 효율을 정량 분석.|
| NLP | Improving Neural Machine Translation with Soft Template Prediction | [Youtube](https://youtu.be/bkJKCvj0LYg)| Machine Translation|소프트 템플릿 예측으로 번역 과정에서 더 자연스러운 구조와 유창성 확보.|
| NLP | Minerva - Solving Quantitative Reasoning Problems with Language Models | [Youtube](https://youtu.be/CMshR8Kpefo)| Question Answering|대형 언어 모델로 수학·정량적 추론 문제를 해결해 높은 정확도 달성.|
| NLP | PaLM: Scaling Language Modeling with Pathways |[Youtube](https://youtu.be/1LDi_TuFy2I) | Language Modeling|Pathways 아키텍처로 대규모 언어 모델을 스케일링해 다양한 태스크 성능 향상.|
| NLP | Unsupervised Neural Machine Translation for Low-Resource Domains via Meta-Learning | [Youtube](https://youtu.be/QDPfIB1MiFg)|Unsupervised Machine Translation|메타학습을 도입해 저자원 도메인에서 무감독 번역 모델의 성능을 향상.|
| NLP | Causal Inference in Natural Language Processing: Estimation, Prediction, Interpretation and Beyond | [Youtube](https://youtu.be/3JBcvwn07x4)|	Causal Inference|NLP 태스크에 인과 추론 기법을 적용해 예측과 해석력을 높이는 연구.|
| NLP | BRIO: Bringing Order to Abstractive SummarizationS | [Youtube](https://youtu.be/25LESLb93ck)|Abstractive Summarization|요약 결과물의 구조 및 일관성을 개선해 더 읽기 쉬운 추상적 요약 생성.|
| NLP | Towards Robust and Reproducible Active Learning using Neural Networks | [Youtube](https://youtu.be/lhzU_ej5pcE)|Active Learning|재현 가능하며 안정적인 능동학습 전략을 제안해 실제 적용성을 높인 연구.|
| NLP | Scaling Instruction-Finetuned Language Models | [Youtube](https://youtu.be/lta-rKYtVbg)|	Instruction Tuning|지시(instruction) 기반 데이터로 파인튜닝 시 모델 스케일업 효과를 분석.|
| NLP | Packed Levitated Marker for Entity and Relation Extraction | [Youtube](https://youtu.be/aiS_iNOOUl8)| Entity and Relation Extraction|마커(marker) 기법을 활용해 엔티티 및 관계를 통합적으로 추출하는 방법 제시.|
| NLP | LLaMA  Review! | [Youtube](https://youtu.be/-lYpL96mIug)|Large Language Models|Meta AI에서 공개한 LLM 시리즈로, 상대적으로 적은 파라미터로도 높은 성능 발휘.|
| NLP | P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks | [Youtube](https://youtu.be/9Ywkk455-Gk)|Prompt Tuning|다양한 크기의 모델에서 프롬프트 튜닝이 풀 파인튜닝에 맞먹는 성능을 보여줌.|
| NLP | MCSE:Multimodal Contrastive Learning of Sentence Embeddings| [Youtube](https://youtu.be/jtpDX9Xvh1A)|Multimodal Sentence Embedding|문장 임베딩을 멀티모달 대조학습으로 확장해 텍스트+이미지 등 통합 표현 학습.|
| NLP | Hyperbolic Image Embeddings | [Youtube](https://youtu.be/a7bOszhprAA)|Image Embeddings|이미지 임베딩을 쌍곡공간(hyperbolic space)에 매핑해 계층적·트리 구조 표현 향상.|
| NLP | LIMA Less Is More for Alignment | [Youtube](https://youtu.be/bw6KDvLPTgU)|Alignment|소량 데이터로도 효율적 정렬(alignment) 가능성을 보인 모델 연구.|
| NLP | Instruct GPT :  Training language models to follow instructions with human feedback | [Youtube](https://youtu.be/vTWV3t_ohQk)|Instruction Tuning|인간 피드백을 활용한 RL 기법으로 GPT가 지시사항을 잘 따르도록 학습.|
| NLP | Iter-RetGen : Enhancing Retrieval augmented Large Language Models with Iterative Retrieval | [Youtube](https://youtu.be/f4Knme-dqP8)|Retrieval-Augmented Generation|반복(iterative) 검색 전략으로 지식 기반 텍스트 생성 품질을 단계적으로 향상.|
| NLP | Textbooks Are All You need | [Youtube](https://youtu.be/YOd7Ukfg7wg)|Language Modeling, Instruction Tuning|교과서 수준의 간단한 텍스트만으로도 대형 언어 모델을 효율적으로 학습할 수 있음을 시사.|
| NLP | Embedding Text in Hyperbolic space | [Youtube](https://youtu.be/6CBj_mwGtK0)|Hyperbolic Embeddings, Representation Learning|쌍곡공간 임베딩으로 텍스트 간 계층적·트리 관계를 효과적으로 학습.|
| NLP | OpineSum : Entailment based self training for abstractive opinion summarization | [Youtube](https://youtu.be/MXugJIFN4qc)|	Opinion Summarization, Entailment|엔테일먼트 기반 셀프 트레이닝으로 사용자 의견 요약 품질을 높이는 기법.|
| NLP | Do Androids Laugh at Electric Sheep  Humor “Understanding” Benchmarks from The New Yorker Caption | [Youtube](https://youtu.be/yCgsCyoJoMg)|Humor Detection, Benchmarking|만화 캡션 등 유머 이해 능력을 평가하는 벤치마크를 제시, AI의 ‘유머 인식’ 성능을 검증.|
| NLP | PUNICA: Multi-Tenant LoRA Serving | [Youtube](https://youtu.be/L8dsadlYc18)|Parameter Efficient Tuning, LoRA Serving|LoRA(저순위 어댑터) 기반 모델을 다중 사용자 환경에서 효율적으로 서비스하는 플랫폼.|
| NLP | From Pretraining Data to Language Models to Downstream Tasks | [Youtube](https://youtu.be/CIYwh6xvaUY)|Language Models, Transfer Learning|사전학습 데이터 선정부터 다운스트림 태스크 적용까지 일관된 언어 모델 학습·분석 파이프라인 제안.|
| NLP | Efficient Memory Management for Large Language Model Serving with pagedAttention | [Youtube](https://youtu.be/l4Xn-jfcBHo)|Memory Optimization, Large Language Models|pagedAttention 기법을 통해 LLM 추론 시 메모리 사용을 효율적으로 관리하는 방법.|
| NLP | Natural Posterior Network Deep Bayesian Uncertainty for Exponential Family Dist | [Youtube](https://youtu.be/aL20zxxC_dI)|Bayesian Learning, Uncertainty Estimation|지수 가족 분포 기반 심층 베이지안 접근으로 모델 불확실성을 더 정확히 추정.|
| NLP | Mamba: Linear Time Sequence Modeling with Selective State Spaces | [Youtube](https://youtu.be/ZEWDBs45nww)|Sequence Modeling, State Space Models|상태 공간을 선택적으로 활용해 선형 시간 복잡도로 시퀀스 모델링 성능 유지.|
| NLP | ChatGPT for Robotics Design Principles and Model Abilities | [Youtube](https://youtu.be/lR0S5uFjk7g)|Human-Robot Interaction, Language Models|ChatGPT를 로보틱스 설계에 적용해 모델 능력과 디자인 원칙을 점검·분석.|
| NLP | Meta in context Learning In large Language Models | [Youtube](https://youtu.be/CXK-SZTC6mI)|Meta-learning, In-Context Learning|대형 언어 모델에서 인컨텍스트 학습을 메타학습 관점으로 해석, 능력을 향상시키는 방법 분석.|
| NLP | Large Language Models Understand and Can be Enhanced by Emotional Stimuli | [Youtube](https://youtu.be/3Jok2PF6sbg)|Emotion-Aware LLM, Sentiment Analysis|감정 자극(Emotional Stimuli)이 대형 언어 모델 이해력 및 성능 향상에 기여함을 제시.|
| NLP | Ignore This Title and HackaPrompt | [Youtube](https://youtu.be/lO-8IeFRj2I)|Prompt Engineering, Adversarial Prompts|‘무시해야 할 제목’ 같은 어드버서리얼 프롬프트를 통해 모델을 교묘히 우회·오류 유발.|
| NLP | AlpacaFarm :  A Simulation Framework for Methods that Learn from Human Feedback | [Youtube](https://youtu.be/xxiWF0EbmCY)|RLHF Simulation, LLM Fine-Tuning|인간 피드백 기반 학습(RLHF)을 시뮬레이션할 수 있는 AlpacaFarm 프레임워크 제안.|
| NLP | Why Can GPT Learn In Context  Language Models Secretly Perform Gradient Desce | [Youtube](https://youtu.be/Y8RUnf4_2VI)|In-Context Learning, Gradient Descent|GPT가 내부적으로 그래디언트 디센트를 모사해 인컨텍스트 학습을 수행한다는 가설.|
| NLP | SmoothQuant :  Accurate and Efficient Post  Training Quantization for Large Language | [Youtube](https://youtu.be/3Y93bKgUyiQ)|Quantization, Model Compression|대형 언어 모델을 후처리 단계에서 양자화(SmoothQuant)하여 정확도·효율 동시 확보.|
| NLP | Generative Representational Instruction Tuning | [Youtube](https://youtu.be/7MZtl45FDXI)|Instruction Tuning, Representation Learning|생성적 표현(Representational)과 지시 기반 튜닝을 결합해 모델 표현력 개선.|
| NLP | Graph of Thought : Solving Elaborate Problems with Large Language Models | [Youtube](https://youtu.be/psVspnBJ9qM)|Complex Reasoning, Graph-based Thinking|	그래프 사고(Graph of Thought) 기법으로 복잡 문제를 대형 언어 모델이 단계적으로 해결.|
| NLP | Hallucination of Multimodal Large Language models | [Youtube](https://youtu.be/6QwMUOzsuaw)|Multimodal LLM, Hallucination|이미지·텍스트 등 멀티모달을 처리하는 LLM에서 발생하는 할루시네이션과 그 완화 방안 연구.|
| NLP | RAPTOR : Recursive Abstractive Processing For Tree Organized Retrieval | [Youtube](https://youtu.be/_EYzMZpwQJM)|Abstractive Summarization, Tree Retrieval|트리 구조 데이터를 재귀적으로 요약해 복잡한 문서 검색 성능을 향상하는 기법.|
| NLP | Chain of Thought Reasoning Without Prompting | [Youtube](https://youtu.be/ae3CVb-3YbA)|Chain of Thought, Zero-Prompt Reasoning|별도 프롬프트 없이도 모델이 단계적 사고 과정을 학습·활용(Chain of Thought) 가능.|
| NLP | Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets | [Youtube](https://youtu.be/daa446K_MGE)|GFlowNet, Graph Optimization|GFlowNets를 활용해 그래프 조합 최적화 문제에서 해 공간을 효율적으로 탐색·해결|
| NLP | MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning | [Youtube](https://youtu.be/_goMNbhRIkI)|Parameter Efficient Tuning, High-Rank Update|저순위 LoRA와 반대로 고순위(MoRA) 업데이트 방식으로 미세조정 효율과 성능을 동시 개선.|
| NLP | Adapting Visual Language Models for Generalizable Anomaly Detection in Medical | [Youtube](https://youtu.be/2D7BNUEuVjY)|Medical Anomaly Detection, Vision-Language Models|비전-언어 모델을 의료 이상 탐지에 활용, 다양한 환경에서 일반화 성능을 높이는 접근.|
| NLP | Gradient Boosting Reinforcement Learning (GBRL) | [Youtube](https://youtu.be/GdgW2D3ffnY)|Reinforcement Learning, Gradient Boosting|그래디언트 부스팅 개념을 RL에 접목해 학습 효율 및 정책 품질을 개선한 기법.|
| NLP | Designing a Dashboard for Transparency and Control of Conversational AI | [Youtube](https://youtu.be/cs_qE2o7u4E)|Conversational AI, Transparency|대화형 AI 작동을 투명하게 모니터·제어 가능한 대시보드를 설계해 사용자 신뢰도 향상.|
| NLP | How Johnny Can Persuade LLMs to Jailbreak Them | [Youtube](https://youtu.be/a7iNwaVzv1Q)|Adversarial Prompting, Model Security|사용자 프롬프트를 통해 모델 정책을 우회(jailbreak)시키는 기법과 방어 시나리오 연구.|
| NLP | LongRoPE : Extending LLM Context Window Beyond 2 Million Tokens | [Youtube](https://youtu.be/6VTRyeZ-HHs)|	Extended Context, Rotary Positional Embedding|RoPE를 확장해 최대 수백만 토큰 이상의 장문을 처리할 수 있는 LLM 구조 제안.|
| NLP | CoVe : Chain of Verification Reduces Hallucination in LLM | [Youtube](https://youtu.be/VXZzfeaH0LU)|	Hallucination Reduction, Verification|	모델 출력의 사실성을 검증(Verification)하는 체인을 추가해 할루시네이션 완화.|
| Vision | ControlNet : Adding Conditional Control to Text to Image Diffusion Model | [Youtube](https://youtu.be/fLWSV9pNpi4)|Text-to-Image, Conditional Diffusion|Diffusion 기반 텍스트→이미지 생성에 조건부 제어를 추가해 생성 결과를 세밀하게 통제.|
| Vision | Interpreting CLIP’s Image Representation via Text Based Decomposition | [Youtube](https://youtu.be/Ms5Ni_0Wrhc)|Vision-Language Models, Representation Analysis|CLIP 임베딩을 텍스트 기반으로 분해·해석해 시각 표현 구조 이해를 개선.|
| Vision | Open-World Semantic Segmentation Including Class Similarity | [Youtube](https://youtu.be/KXw9OyZ68IU)|Open-World Segmentation, Class Similarity|클래스를 모르는 오픈월드 세그멘테이션 상황에서 클래스 유사도 등을 고려해 분할 정확도 향상.|
| Vision | Dynamic Sparse Voxel Transformer with Rotated Sets (DSVT) | [Youtube](https://youtu.be/wQs_iBT96Os)|3D Object Detection, Sparse Transformer|3D 점구름(voxel)에서 회전된 셋을 활용, 희소성·동적성을 겸비한 Transformer로 물체 탐지.|
| Vision | Is ImageNet Worth 1 Video? | [Youtube](https://youtu.be/j-u8U2CgXlg)|Video Pretraining, ImageNet Alternative|단 하나의 긴 동영상이 ImageNet에 상응하는 사전학습 효과를 낼 수 있는지 실험적으로 검증.|
| Vision | SAM2 | [Youtube](https://youtu.be/TkYZgoPzxQQ)|Semantic Segmentation, SAM|Segment Anything Model(SAM)을 확장 혹은 개선하여 세분화 정확도 향상 모색. 심지어 동영상|
| Vision | SparK : DESIGNING BERT FOR CONVOLUTIONAL NETWORKS | [Youtube](https://youtu.be/EKj_DHdCkhs)|ConvNets, BERT-like Architecture|CNN에 BERT 구조를 접목해 시각 특징 학습을 강화하는 SparK 모델 제안.|
| Vision | Diffusion-based Semantic-Discrepant Outlier Generation for Out-of-Distribution Detection | [Youtube](https://youtu.be/CKgzGtL9KRs)|OOD Detection, Diffusion Models|정상 데이터와 다른(semantic discrepancy) 샘플을 생성해 OOD 탐지 성능 강화.|
| Vision | Tri Perspective View for Vision  Based 3D Semantic Occupancy Prediction | [Youtube](https://youtu.be/EZiposPneps)|3D Occupancy, Semantic Prediction|세 가지 관점(Tri Perspective)으로 3D 공간의 점유 및 의미 정보를 예측하는 접근.|
| Vision | UniCLIP Unified Framework for Contrastive Language Image Pre training | [Youtube](https://youtu.be/CIYwh6xvaUY)|CLIP, Contrastive Pretraining|언어–이미지 대조학습을 하나로 통합한 프레임워크로 범용 비전-언어 표현 학습 지원.|
| Vision | BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation | [Youtube](https://youtu.be/UAGOB3s1J3c)|Sensor Fusion, 3D Detection|라이다·카메라 등 센서 정보를 Bird’s-Eye View로 융합해 다중 태스크를 동시 처리.|
| Vision | SemiVL: Semi-Supervised Semantic Segmentation with Vision-Language Guidance | [Youtube](https://youtu.be/_eKw6ELOumI)|Semi-Supervised Segmentation, Vision-Language|비전-언어 정보를 활용해 반지도(세미 슈퍼바이즈) 세그멘테이션 성능을 높임.|
| Vision | YDTR Infrared and Visible Image Fusion via Y Shape Dynamic Transformer | [Youtube](https://youtu.be/dRjw4kVvQjQ)|mage Fusion, Transformer|적외선·가시광 이미지를 Y자형 동적 Transformer로 융합해 시각적 품질 향상.|
| Vision | STABLEREP : Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners | [Youtube](https://youtu.be/g-v0whlAQjw)|Synthetic Data, Representation Learning|텍스트→이미지 합성 데이터를 활용해도 강력한 시각 표현 학습을 달성할 수 있음.|
| Vision | Diffusion models Beat GANs on Image Synthesis | [Youtube](https://youtu.be/Y3kmXf9aT78)|Image Synthesis, Diffusion vs. GAN|확산(diffusion) 모델이 이미지 생성 품질 면에서 GAN을 능가한다는 연구 결과.|
| Vision | DOMINO   Discovering Systematic Errors with Cross Modal Embeddings | [Youtube](https://youtu.be/cnD2-B74mVo)|Error Analysis, Cross-Modal Embeddings|모델의 체계적 에러를 멀티모달 임베딩을 통해 탐색·분류해 개선 여지 확인.|
| Vision | Emergent Complexity and Zero shot Transfer via Unsupervised Environment Design | [Youtube](https://youtu.be/5qxwfWznNEc)|Unsupervised Environment Design, Zero-Shot Transfer|학습 환경을 자동 설계해 에이전트가 예측 못한 태스크로 제로샷 전이 가능하게 만듦.|
| Vision | RT-DETR : DETRs Beat YOLOs On Real time Object Detection | [Youtube](https://youtu.be/We8LWF6QaOI)|Object Detection, Transformer-based DETR|DETR 계열 모델이 실시간 객체 검출(throughput)에서 YOLO와 경쟁·상회하는 결과 보고.|
| Vision | Masked Auto encoders Are Scalable Vision Learners | [Youtube](https://youtu.be/kOwwPac76RA)|MAE, Self-Supervised Learning|이미지 일부를 마스킹 후 복원하는 MAE 기법으로 대규모 시각 표현 학습의 효과를 입증.|
| Vision | LLM-grounded Diffusion | [Youtube](https://youtu.be/mkxrmr5nwOo)|Vision-Language Models, Diffusion|대형 언어 모델(LLM)을 접목해 텍스트 이해력을 강화한 이미지 생성.|
| Vision | Depth Anything Unleashing the Power of Large Scale Unlabeled Data | [Youtube](https://youtu.be/y985cAWRStk)|Depth Estimation, Unlabeled Data|	대규모 비라벨 데이터에서 깊이(depth) 정보를 학습해 다양한 비전 태스크로 확장 가능.|
| Vision | Object Lab | [Youtube](https://youtu.be/_hnpFhexqiU)|Object Discovery, 3D Reconstruction|	객체 탐지·3D 재구성을 결합한 ‘Object Lab’ 환경을 통해 직관적 오브젝트 연구.|
| Vision | YOLO | [Youtube](https://www.youtube.com/watch?v=Ae-p7QVOdbA&t=285s) <br> [paper](https://arxiv.org/pdf/1506.02640.pdf) |Object detection|Object detection|	이미지 내 객체를 단일 신경망으로 빠르게 검출하는 혁신적 원스테이지 모델.|
| Vision  | YOLO-v2 |[Youtube](https://www.youtube.com/watch?v=9FiGYp6khxo&t=8s) |Object detection |Object detection	|YOLO 대비 앵커 박스·배치 정규화 등 개선으로 정확도·속도 향상에 성공.|
| Vision  | Resnet | [Youtube](https://www.youtube.com/watch?v=JI5kXF_OUkY&t=125s) <br> [paper](https://arxiv.org/pdf/1512.03385.pdf) |Image classification|Image classification|	심층 신경망에서 기울기 소실 문제를 잔차(residual) 블록으로 해결해 학습 안정화.|
| Vision  | GAN | [Youtube](https://www.youtube.com/watch?v=UZpuIG1eF8Y&t=147s) |Generative Adversarial Networks, Generative Models|Generative Adversarial Networks, Generative Models	|생성자와 판별자가 경쟁적으로 학습해 고품질 이미지를 생성하는 혁신적인 생성 모델(GAN).|
| Vision | Image Style Transfer Using CNN | [Youtube](https://www.youtube.com/watch?v=8jS0xxslTco&t=905s) | Convolutional Neu- ral Network|	CNN을 활용해 콘텐츠 이미지는 유지하고, 스타일 이미지를 결합하여 새로운 스타일 이미지 생성.|
| Vision  | SINGAN | [Youtube](https://www.youtube.com/watch?v=pgYIuA4O95E) | Image Generation, Image Manipulation|단일 이미지만으로 멀티 스케일 GAN을 학습해 다양한 이미지 변환·조작 수행 가능.|
| Vision  | FCN | [Youtube](https://www.youtube.com/watch?v=_52dopGu3Cw) | Semantic Segmentation Models |완전 합성곱 네트워크(FCN)로 픽셀 단위 세그멘테이션 수행, 보다 정확한 분할 실현.|
| Vision | DeepLabV3| [Youtube](https://youtu.be/TjHR9Z9iNLA)| Semantic Segmentation Models|팽창(a trous) 컨볼루션과 ASPP로 객체 크기 변화에 대응하며 세밀한 분할 가능.|
| Vision | Unet | [Youtube](https://www.youtube.com/watch?v=evPZI9B2LvQ&t=9s) <br> [paper](https://arxiv.org/pdf/1505.04597v1.pdf)| Cell Segmentation, Multi-tissue Nucleus Segmentation |인코더-디코더 구조로 의료영상과 같은 소량 데이터 환경에서도 높은 분할 정확도.|
| Vision | CyCADA | [Youtube](https://youtu.be/DODYdEwebTg)| Domain Adaptation, Semantic Segmentation|생성적 적대 학습을 통해 소스·타겟 도메인 차이를 줄여 세그멘테이션 성능 향상.|
| Vision | D-SNE | [Youtube](https://youtu.be/OJe9SgS-GM8)| Domain Adaptation|	t-SNE 시각화를 기반으로 도메인 불일치를 줄여 분류·세그멘테이션 성능 개선.|
| Vision | Faster-RCNN| [Youtube](https://youtu.be/HmJWvwIpW5g)| Real-Time Object Detection, Region Proposal|R-CNN 계열을 개선해 영역 제안(Region Proposal) 과정을 통합, 속도·정확도 동시 향상.|
| Vision | Weakly Supervised Object Detection With Segmentation Collaboration| [Youtube](https://youtu.be/qvWf0aIqaLE)| General Classification, Image Classification|이미지 레벨 라벨만으로 객체를 검출하고, 분할 협업으로 정확도 보완하는 약지도 방식.|
| Vision | Don't Judge an Object by Its Context: Learning to Overcome Contextual Bias| [Youtube](https://youtu.be/xTm1gWWSEHM) |object classification|주변 문맥에 의존하는 편향을 줄이고 객체 자체 특징으로 분류 정확도를 높이는 연구.|
| Vision | data efficient image recognition with contrastive predictive coding| [Youtube](https://youtu.be/-LzFAqOnfTo)| General Classification, Object Detection|적은 데이터 상황에서 대조 예측 부호화(CPC)로 효율적인 시각 특성 학습 달성.|
| Vision | Deep Feature Consistent Variational Autoencoder| [Youtube](https://youtu.be/Iy0zCVZBO_A)| Style Transfer|VAE에 딥 특징 일관성 조건을 추가해 스타일 전이 품질·다양성 개선.|
| Vision | Attention Branch Network: Learning of Attention Mechanism for Visual Explanation| [Youtube](https://youtu.be/8BAGbC0HCVg)| Decision Making, Image Classification|분기된 어텐션 브랜치로 시각적 중요 영역을 강조, 해석 가능성과 분류 정확도 동시 향상.|
| Vision | RELATION-SHAPE CONVOLUTIONAL NEURAL NETWORK FOR POINT CLOUD ANALYSIS| [Youtube](https://youtu.be/E_odPJHLW7Y)|point cloud|점구름 데이터에서 물체 형태·상호 관계를 학습하기 위한 CNN 기반 구조 제안.|
| Vision | EfficientNet| [Youtube](https://youtu.be/Vy0BvuFSNxQ) |CNN| 너비·깊이·해상도를 동시에 확장해 효율적으로 정확도 향상(Scaling)을 달성한 CNN 모델.|
| Vision | Deep Clustering for Unsupervised Learning of Visual Features| [Youtube](https://youtu.be/cCwzxVwfrgM)| Deep Clustering, Image Clustering|비지도 클러스터링과 이미지 특징 학습을 결합해 시각 표현 학습 정확도 향상.|
| Vision | Boosting Few-shot visual learning with self-supervision| [Youtube](https://youtu.be/6ZrXjdMfqxk)| Few-Shot Learning, Self-Supervised Learning|자가 지도 방식으로 특성 품질을 향상, 극소량(Few-shot) 샘플에서도 높은 정확도 가능.|
| Vision | Rethinking Pre-training and Self-training| [Youtube](https://youtu.be/d8EDoHDEgvI) | Data Augmentation, Object Detection|사전학습 + 셀프 트레이닝 전략 재검토로 객체 검출 등에서 데이터 효율성 향상.|
| Vision | BYOL : Bootstrap Your Own Latent| [Youtube](https://youtu.be/BuyWUSPJicM) | Representation Learning, Self-Supervised Image Classification|서로 다른 네트워크 간 상호 학습(교사·학생)으로 점진적 표현 학습 향상.|
| Vision | Deep Image Prior| [Youtube](https://youtu.be/wvc_JX4WUHo) |Image Denoising, Image Inpainting|네트워크 구조 자체로 이미지를 자연스럽게 복원하도록 유도, 별도 사전학습 불필요.|
| Vision | Object-Centric Learning with Slot Attention| [Youtube](https://youtu.be/dMGFQ_ISdFg) |Object Discovery|슬롯 어텐션 기법으로 이미지 내 객체들을 분리·학습해 객체 중심 표현을 확보.|
| Vision | Yolo V4| [Youtube](https://youtu.be/D6mj_T8K_bo) | Data Augmentation, Real-Time Object Detection|Mosaic, IoU-aware 등 다양한 기법을 결합해 실시간 검출 성능을 대폭 향상한 YOLO 후속.|
| Vision | Dynamic Routing Between Capsules|[Youtube](https://youtu.be/aH7Hn-Ca_uk) | Image Classification|캡슐 간 동적 라우팅(Dynamic Routing)으로 시각적 계층 구조를 보존하며 분류 정확도 향상.|
| Vision | Semi-Supervised Classification with Graph Convolutional Network|[Youtube](https://youtu.be/Ft2Q8WQ8ETM) | Classification, Document Classification|그래프로 연결된 노드 관계를 활용해 반지도 분류·문서 분류 정확도 향상.|
| Vision | Generative Pretraining from Pixels|[Youtube](https://youtu.be/QC9VWEv7qrw) | Representation Learning, Self-Supervised Image Classification|픽셀 단위에서 생성적 사전학습 후, 다양한 시각 태스크로 전이 가능.|
| Vision | MaskFlownet | [Youtube](https://youtu.be/8J3_BeVoCUg) | Feature Matching|Optical Flow 예측과 마스킹을 결합해 정교한 특징 매칭을 수행하는 네트워크.|
| Vision | Adversarial Robustness through Local Linearization| [Youtube](https://youtu.be/-h2W5-A8qNU) |Adversarial Defense, Adversarial Robustness|적대적 공격 주변에서 모델을 국소 선형화하여 내재적 강건성을 확보.|
| Vision | Locating Objects Without Bounding Boxes| [Youtube](https://youtu.be/TkcRI31v5_I) | Object Localization|바운딩 박스 라벨 없이 객체 위치를 추론해 라벨 비용 절감 및 유연성 확보.|
| Vision | Training data-efficient image transformers & distillation through attention| [Youtube](https://youtu.be/LYYxv9mv5qw) | Document Image Classification, Document Layout Analysis|트랜스포머와 어텐션 기반 지식 증류를 통해 데이터 효율적으로 문서 인식 성능 향상.|
| Vision | What Makes Training Multi-modal Classification Networks Hard?| [Youtube](https://youtu.be/ZjDRVgA9F1I) | Action Classification, Action Recognition|멀티모달 분류 네트워크 학습 시 발생하는 편향·난관을 분석해 해법 모색.|
| Vision | Meta-Transfer Learning for Zero-Shot Super-Resolution| [Youtube](https://youtu.be/lEqbXLrUlW4) | Image Super-Resolution, Meta-Learning|메타 전이학습을 통해 초해상도(SSR) 문제에서 제로샷 환경도 효과적으로 대응.|
| Vision | Patch SVDD: Patch-level SVDD for Anomaly Detection and Segmentation| [Youtube](https://youtu.be/RrGVDQLEnq4) | Anomaly Detection, Self-Supervised Learning|국소(패치) 단위 SVDD 적용으로 미세한 이상 탐지·세그멘테이션 가능.|
| Vision | Style GAN| [Youtube](https://youtu.be/YGQTzYNIL0s) | Disentanglement, Image Generation|스타일 공간을 제어해 이미지 생성의 다채로운 변형·속성 조절 가능.|
| Vision | High-Performance Large-Scale Image Recognition Without Normalization| [Youtube](https://youtu.be/HP4evlugOIo) | Image Classification|정규화 없이도 대규모 이미지 분류에서 높은 성능을 내는 학습 기법 제안.|
| Vision | Focal Loss for Dense Object Detection| [Youtube](https://youtu.be/d5cHhLyWoeg) |Object Detection| 클래스 불균형 심한 객체 검출 상황에서 포컬 로스로 성능 극대화.|
| Vision | Editing in Style : Uncovering the Local Semantics of GANs| [Youtube](https://youtu.be/2Gx3_0xpNvM) | Dense Object Detection, Long-tail Learning|GAN 내부 로컬 스타일 코드를 편집해 이미지 수정·생성 자유도를 확보.|
| Vision | EfficientNet 2| [Youtube](https://youtu.be/uLKqMbOA_vU) | Convolutional Neural Networks, Image Models|EfficientNet를 확장한 모델로, 크기·성능 균형을 한층 향상.|
| Vision | Style Clip| [Youtube](https://youtu.be/5FwzEP3bYLg) |Image Manipulation|텍스트 프롬프트와 GAN을 결합해 직관적으로 이미지 스타일 조절 가능.|
| Vision | Swin Transformer| [Youtube](https://youtu.be/L3sH9tjkvKI) | Image Models, Vision Transformers|스윈(Window) 어텐션으로 고해상도 이미지를 효율적으로 처리하는 비전 트랜스포머.|
| Vision | NBDT: Neural-backed Decision Tree| [Youtube](https://youtu.be/MdDAug75J6s) |Neural-Backed Decision Trees|신경망으로 학습한 계층적 트리 구조로 결정 과정을 시각적으로 해석 가능.|
| Vision | EfficientDet | [Youtube](https://youtu.be/Mq4aqDgZ2bI) | Semantic Segmentation Models, One-Stage Object Detection Models, Object Detection Models|BiFPN과 EfficientNet 백본으로 높은 정확도와 빠른 속도를 동시에 달성한 검출 모델.|
| Vision | MLP - MIXER: An all-MLP Architecture for Vision | [Youtube](https://youtu.be/L3vEetyNG_w) | Image Classification|어텐션 없이도 채널·공간 MLP만으로 이미지 분류를 수행, 단순 구조로도 준수한 성능 달성.|
| Vision | You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection | [Youtube](https://youtu.be/hqRSw_Bu24w) |Object Detection|	비전 트랜스포머를 한 번의 시퀀스 처리로 객체 검출에 적용, 효율적 구조 제안.|
| Vision | Hierarchical Long-term Video Frame Prediction without Supervision | [Youtube](https://youtu.be/q15QO9LXYlI) |Video Prediction|비지도 학습으로 장기 비디오 프레임을 계층적으로 예측·생성.|
| Vision | Closed-Form Factorization of Latent Semantics in GANs | [Youtube](https://youtu.be/TKmIalR5x8I) |Image Generation, Image Manipulation|GAN 잠재공간을 닫힌 형태로 분해해 특정 속성만 선택적으로 수정·생성 가능.|
| Vision | You Only Learn One Representation: Unified Network for Multiple Tasks |[Youtube](https://youtu.be/bQgQkhaGZG8) |Multi-Task Learning, Real-Time Object Detection| 여러 비전 태스크를 단일 표현(Representation)으로 처리하는 통합 네트워크 제안, 실시간 객체 검출 등 동시 수행.|
| Vision | StyleSpace Analysis |[Youtube](https://youtu.be/JcIe5U5PmLQ) |Image Generation|스타일 공간(StyleSpace)을 분석해 이미지를 세부적으로 조작·생성하는 방법론 연구.|
| Vision | Representative graph neural network |[Youtube](https://youtu.be/z-UUq8x1oRw) |Representative Graph, Dynamic Sampling, Semantic Segmentation|그래프 신경망에 대표성(Representative) 개념을 도입해 동적 샘플링·의미론적 세그멘테이션 성능 개선.|
| Vision | YOLOX |[Youtube](https://youtu.be/N2rLSzEqqI8) |One-Stage Object Detection Models|YOLO 계열을 확장·개선한 원스테이지 검출 모델로 앵커 프리(anchor-free) 구조가 특징.|
| Vision | Joint Contrastive Learning with Infinite Possibilities |[Youtube](https://youtu.be/0NLq-ikBP1I) |Contrastive Learning|대조학습을 확장해 ‘무한’한 양상의 표현 학습 가능성을 모색, 다양한 뷰·도메인 통합 학습.|
| Vision | Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation |[Youtube](https://youtu.be/2886fuyKo9g) | Image Classification, Neural Architecture Search|자동화된 계층형 아키텍처 탐색을 통해 세그멘테이션 모델을 최적 설계, 이미지 분류 등에도 적용 가능.|
| Vision | Explaining in style training a gan to explain a classifier in stylespace |[Youtube](https://youtu.be/pkbLrLSDQ9Q) | Image Classification|StyleSpace 기반으로 분류기의 결정 경로를 GAN으로 시각화·설명하는 접근.|
| Vision | End-to-End Semi-Supervised Object Detection with Soft Teacher |[Youtube](https://youtu.be/7gza1VFjb0k) | Instance Segmentation, Object Detection|반지도 학습에서 Teacher 모델을 소프트하게 결합해 물체 검출·인스턴스 분할 성능 향상.|
| Vision | Understanding Dimensional Collapse in Contrastive Self Supervised Learning |[Youtube](https://youtu.be/dO-gD54OlO0) | Contrastive Learning, Learning Theory|대조학습 시 발생하는 표현 차원의 붕괴 현상을 이론적으로 분석하고, 이를 완화하는 방안 제시.|
| Vision | Encoding in Style: a Style Encoder for Image-to-Image Translation |[Youtube](https://youtu.be/2QXQZHvx6Ds) | Conditional Image Generation, Face Generation|스타일 인코더를 도입해 얼굴·이미지 변환 시 개별 스타일 요소를 보다 정확히 제어.|
| Vision | Detection in Crowded Scenes: One Proposal, Multiple Predictions |[Youtube](https://youtu.be/LPC4m66YZfg) | Object Detection, Pedestrian Detection|혼잡한 장면에서 하나의 영역 제안으로 여러 객체(보행자 등)를 예측해 검출 정확도 높임.|
| Vision | A Normalized Gaussian Wasserstein Distance for Tiny Object Detection |[Youtube](https://youtu.be/eGKlg4sZ0Zw) |Object Detection|작은 물체 검출을 위해 정규화된 가우시안 Wasserstein 거리를 활용, 좌표 회귀를 안정화.|
| Vision | Siamese Neural network for one-shot image recognition |[Youtube](https://youtu.be/SthmLerAeis) | One-Shot Learning|쌍둥이(Siamese) 네트워크로 극소량 샘플(1샷)만으로 이미지 인식 가능.|
| Vision | Grounded Language-Image Pre-training |[Youtube](https://youtu.be/krP3t31fWvI) |Object Detection    Phrase Grounding|이미지와 언어를 연결해 사물 검출·어구(phrase) 정합을 사전학습으로 습득, 다양한 멀티모달 태스크에 적용.|
| Vision | Transfer Learning for Pose Estimation of Illustrated Characters |[Youtube](https://youtu.be/wo3Ob174AfY) | Activity Recognition, Pose Estimation|일러스트(만화) 캐릭터 포즈 추정에 기존 인체 포즈 모델을 전이학습, 동작 인식 정확도 개선.|
| Vision | Sparse R-CNN: End-to-End Object Detection with Learnable Proposals |[Youtube](https://youtu.be/EvPMwKALqvs) |Object Detection, Object Recognition|사전에 학습된(proposal) 박스를 직접 예측·조정하는 방식으로, RCNN 단계를 간소화·속도↑|
| Vision | NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis |[Youtube](https://youtu.be/Mk0y1L8TvKE) |Neural Rendering, Novel View Synthesis|신(scene)을 연속적 방사장(Radiance Field)으로 표현해 다양한 시점 뷰 합성(3D 렌더링) 가능.|
| Vision | HRnet: Deep High-Resolution Representation Learning for Human Pose Estimation |[Youtube](https://youtu.be/GmANBttiHx4) |Pose Estimation, Human Pose|고해상도 특징 맵 유지 구조로 인체 포즈 추정 등에서 높은 정밀도 발휘.|
| Vision | MobileViT : Light-weight, general-purpose, and Mobile-friendly Vision Transformer |[Youtube](https://youtu.be/1vZ2wEsUdZc) |Mobile Vision Transformer, Lightweight Architecture|	모바일 환경에서도 효율적으로 동작하도록 설계된 경량 비전 트랜스포머 구조 제안.|
| Vision | Effectively Leveraging Attributes for Visual Similarity |[Youtube](https://youtu.be/FrdtIqyVhbc) |Attribute-based Image Retrieval, Visual Similarity| 시각적 유사도 계산 시 속성(attributes)을 적극 활용해 검색·추천 품질 향상.|
| Vision | Hard Negative Mixing for Contrastive learning |[Youtube](https://youtu.be/0hiHFJXcVvY) |Contrastive Learning, Hard Negative Mining|	어려운 음성 샘플(하드 네거티브)을 혼합함으로써 대조학습 임베딩을 더욱 견고하게 학습.|
| Vision | HOTR : Human-Object Interaction Detection with Transformer |[Youtube](https://youtu.be/OMTnuaemG_A) |Human-Object Interaction Detection, Transformer|사람과 사물의 상호작용(HOI)을 Transformer로 검출하여 행동 이해·분석에 활용.|
| Vision | SSD: A UNIFIED FRAMEWORK FOR SELF-SUPERVISED OUTLIER DETECTION |[Youtube](https://youtu.be/37S7YuWDbng) |Self-Supervised Outlier Detection|	자가 지도 방식으로 이상치(outlier) 검출을 수행하는 통합 프레임워크.|
| Vision | StyleHEAT : One-Shot High-Resolution Editable Talking Face Generation via Pre-trained StyleGAN |[Youtube](https://youtu.be/FuUXwPffnrw) |Talking Face, One-Shot Generation|스타일GAN을 활용해 단일 얼굴 이미지만으로도 고해상도·편집 가능한 페이스 영상 생성 가능.|
| Vision | Self-Supervised Transformers for Unsupervised Object Discovery using Normalized Cut |[Youtube](https://youtu.be/JCEK5nD4MKM) |Unsupervised Object Discovery, Self-Supervised Transformers|	정규화 컷(NCut)과 트랜스포머 결합으로 라벨 없이 객체 분할 정확도 향상.|
| Vision | Transfusion: Understanding Transfer Learning for Medical Imaging |[Youtube](https://youtu.be/b2izifKXYA8) |Medical Imaging, Transfer Learning|의료 영상에 특화된 전이학습 전략을 분석, 기존 대비 정확도·효율 개선.|
| Vision | UCTransNet |[Youtube](https://youtu.be/SFwsc7s_Bww) |Medical Segmentation, U-Net + Transforme|	U-Net과 트랜스포머를 결합해 의료 영상 분할에서 높은 정확도 달성.|
| Vision | A Simple Baseline for Semi-supervised Semantic Segmentation with Strong Data Augmentation |[Youtube](https://youtu.be/-ShKE82TSYI) |Semi-supervised Semantic Segmentation, Data Augmentation|	강력한 증강 기법을 적용해 반지도 세그멘테이션의 기본 성능을 크게 끌어올린 방법론.|
| Vision | Vision Transformer with Deformable Attention |[Youtube](https://youtu.be/hU7gP3u-tLQ) |Deformable Attention, Vision Transformer|	비정형(Deformable) 어텐션으로 스케일 변화가 큰 물체 인식 성능 개선.|
| Vision | More ConvNets in the 2020s: Scaling up Kernels Beyond 51x51 using Sparsity |[Youtube](https://youtu.be/gK40oDVUZ9Q) |Sparse Large-Kernel Conv, Scalable CNN|	51×51을 넘는 초대형 커널을 희소 기법으로 처리해 CNN의 확장성과 정확도 높임.|
| Vision | PseCo: Pseudo Labeling and Consistency Training for Semi-Supervised Object Detection |[Youtube](https://youtu.be/C-NVH6StFQw) |Semi-Supervised Object Detection, Consistency Regularization|	반지도 학습에서 의사 라벨링과 일관성 훈련을 결합, 적은 라벨로도 객체 검출 성능 향상.|
| Vision | MetaFormer is Actually What You Need for Vision |[Youtube](https://youtu.be/ThCkRzh9Ohw) |Transformer Variation, MetaFormer|	“어텐션”보다 중요한 건 메타포머 구조라 주장, 다양한 비전 태스크에 적용 가능.|
| Vision | An Image is Worth 16x16 Words:Transformers for Image Recognition at Scale |[Youtube](https://youtu.be/MvZ2wzghbCg) |Vision Transformer (ViT), Large-Scale Recognition|	이미지 패치를 토큰화한 대규모 트랜스포머로 분류, ViT의 효시가 된 연구.|
| Vision | BigDatasetGAN: Synthesizing ImageNet with Pixel-wise Annotations |[Youtube](https://youtu.be/VgYlBHXKAs0) |GAN-based Data Synthesis, Pixel-wise Annotation|대규모 ImageNet 스타일 이미지를 픽셀 단위 주석까지 합성, 데이터 증강에 활용.|
| Vision | High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion) |[Youtube](https://youtu.be/7fBQDaJkcSU) |Diffusion Models, High-Resolution Image Generation|	잠재 공간에서 확산 모델을 적용해 고해상도 이미지를 생성, Stable Diffusion 아이디어의 핵심.|
| Vision | Proper Reuse of Image Classification Features Improves Object Detection |[Youtube](https://youtu.be/Ikg5Mx3ITh4) |Feature Reuse, Object Detection|	분류용으로 학습된 특징을 적절히 재사용하면 검출 성능을 크게 향상시킬 수 있음을 입증.|
| Vision | Voxel Field Fusion for 3D Object Detection |[Youtube](https://youtu.be/NWAciZYUiXA) |3D Object Detection, Voxel Fusion|	다중 뷰(voxel field) 정보를 융합해 점구름 기반 3D 검출 정확도 향상.|
| Vision | MedSefDiff: Medical Image Segmentation with Diffusion Probabilistic Model |[Youtube](https://youtu.be/gY3iFOIRs48) |Medical Image Segmentation, Diffusion Models|	확산(디퓨전) 확률 모델을 의료 영상 세그멘테이션에 적용해 정밀도 높임.|
| Vision | Pix2Seq |[Youtube](https://youtu.be/nLazWsQ5Opk) |Unified Generation Approach, Object Detection|	객체 검출을 시퀀스 생성 문제로 전환하여 단일 프레임워크에서 처리.|
| Vision | Self-Supervised Learning based on Heat Equation |[Youtube](https://youtu.be/oouLzu0sJfY) |Self-Supervised Representation, Heat Equation|	열 방정식 기반 자가 지도 접근으로 안정적인 특징 학습을 구현.|
| Vision | Cliff Diving : Exploring Reward Surfaces in Reinforcement Learning Environments |[Youtube](https://youtu.be/7OLrLrTYbm8) |Reinforcement Learning, Reward Surfaces|	Cliff Diving 환경에서 보상 지형(Reward Surface)을 분석해 RL 동작 특성 파악.|
| Vision | WHAT DO VISION TRANSFORMERS LEARN?A VISUAL EXPLORATION |[Youtube](https://youtu.be/lU4hyvXUJhw) |Vision Transformer, Interpretability|	ViT 내부에서 어떤 시각 패턴을 학습하는지 시각화·해석.|
| Vision | InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions |[Youtube](https://youtu.be/e5lDg9O4FIQ) |Foundation Models, Deformable Convolution|	대규모 비전 기본 모델에 변형가능 컨볼루션을 적용해 정확도·효율 동시 개선.|
| Vision | Cut and Learn for Unsupervised Object Detection and Instance Segmentation |[Youtube](https://youtu.be/lpDY2uhJJoc) |Unsupervised Detection, Instance Segmentation|	라벨 없이 객체를 잘라내(cut) 학습하며 동시에 객체 감지와 인스턴스 세그멘테이션을 달성.|
| Vision | The Forward-Forward Algorithm: Some Preliminary Investigations |[Youtube](https://youtu.be/vSzp1nPvYSw) |Forward-Forward Training, Alternative Backprop|	역전파 없이 전방(Forward) 단계만 반복해 학습하는 대안적 기법 초기 연구.|
| Vision | Visual Prompt Tuning |[Youtube](https://youtu.be/bVOk-hSYyZw) |Prompt Tuning, Vision Transformers|	기존 비전 모델에 프롬프트(시각 템플릿)를 추가해 특정 태스크로 효율적 전이.|
| Vision | Dataset Distillation by Matching Training Trajectories |[Youtube](https://youtu.be/w8zB77WaT0g) |Dataset Distillation, Efficient Retraining|모델 학습 과정을 모사·압축한 데이터셋으로 빠른 재학습·전이 가능.|
| Vision | YOLO v6 |[Youtube](https://youtu.be/UOGMOLe0xtc) |One-Stage Object Detection, YOLO Series|	YOLO 계열 최신 버전으로, 경량화·정확도 모두 개선한 객체 검출 프레임워크.
| Vision | InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning |[Youtube](https://youtu.be/6Zxs7srpuK8) |Vision-Language, Instruction Tuning|	시각+언어 모델에 지시문(Instruction)을 학습해 다목적 태스크 수행 능력 강화.|
| Vision | ImageBind One Embedding Space To Bind Them All |[Youtube](https://youtu.be/AeJxJBPBzeo) |Multi-modal Embedding, Universal Representation|	텍스트·오디오·이미지 등 여러 모달을 단일 임베딩 공간으로 통합해 범용 표현.|
| Vision | Learning to Generate Text grounded Mask for Open world Semantic Segmentation from Only Image-Text Pairs |[Youtube](https://youtu.be/jEpk8h-Grw8) |Open-world Segmentation, Vision-Language|	이미지-텍스트 쌍만으로 세그멘테이션 마스크를 생성, 오픈월드 환경에 적용 가능.|
| Vision | Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold |[Youtube](https://youtu.be/2oLWZe9M1LU) |GAN-based Image Editing, Interactive Manipulation|	GAN 잠재공간에서 포인트 드래그 방식으로 직관적 이미지 편집·변형.|
| Vision | Segment Anything in High Quality |[Youtube](https://youtu.be/pUedVuvkvnU) |High-Quality Segmentation, SAM|	Segment Anything 모델을 개선해 고화질·정확도 높은 세분화 제공.|
| Vision | Multi Domain Learning for Motion Magnification |[Youtube](https://youtu.be/Sg7_sXMnRLo) |Motion Magnification, Multi-Domain Learning|	다양한 도메인에서 미세한 움직임을 증폭해 관찰하도록 학습, 의료·산업 등 활용 가능.|
| Vision | 3D Gaussian Splatting for Real Time Radiance Field Rendering |[Youtube](https://youtu.be/gqJCWyYPtXQ) |NeRF Rendering, Real-time|	3D 점 클라우드에 가우시안 스플랫을 적용, 실시간 방사장(NeRF) 렌더링 구현.|
| Vision | VoxelNet End-to-End Learning for Point Cloud Based 3D Object Detection |[Youtube](https://youtu.be/Q29eDs4nuWk) |3D Object Detection, Voxel-based Networks|	VoxelNet 구조로 점구름(포인트 클라우드) 입력을 직접 학습, 3D 검출 정확도 개선.|
| Vision | DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation |[Youtube](https://youtu.be/jq85UXiJEXk) |Text-to-Image Diffusion, Subject-driven Generation|	텍스트→이미지 확산 모델을 특정 주제나 스타일에 맞춰 파인튜닝, 맞춤형 이미지 생성 가능.|
| Vision | MIC Masked Image Consistency for Context Enhanced Domain Adaptation |[Youtube](https://youtu.be/IgwpPkSpwUg) |Domain Adaptation, Masked Consistency|	마스킹된 이미지를 활용해 도메인 적응 시 맥락 정보 일관성을 유지, 성능 강화.|
| Recommend System | Matrix Factorization Technique for Recommender System | [Youtube](https://www.youtube.com/watch?v=Z49JNxS4vsc&t=260s) <br> [paper](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)| Recommendation system |사용자-아이템 행렬 분해 기법으로 대규모 협업필터링 구현, 넷플릭스 등에서 성능 입증.|
| Recommend System| Collaborative Filtering for Implicit Feedback Dataset | [Youtube](https://www.youtube.com/watch?v=ePvzTeLOBi4&t=6s) |recommender systems, Collaborative Filtering|묵시적 피드백(조회·클릭) 기반 사용자 취향 예측을 협업필터링으로 처리하는 방법론.|
| Speech | A comparison of S2S models for speech recognition | [Youtube](https://www.youtube.com/watch?v=fltpFsNL8TA&t=463s)  <br> [paper](https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0233.PDF) | Speech Recognition|음성인식에서 여러 S2S(seq2seq) 모델을 비교·분석하여 장단점 파악 및 성능 비교.|
| Fundamental | RAdam | [Youtube](https://www.youtube.com/watch?v=_F5_hgX_lSE) <br> [blog](https://hiddenbeginner.github.io/deeplearning/2019/09/22/optimization_algorithms_in_deep_learning.html) <br> [paper](https://arxiv.org/pdf/1908.03265.pdf)| Regularization,  Stochastic Optimization|Adam의 변동성 문제를 레디언스(RA) 항으로 보정해 안정적 훈련 가능.|
| Fundamental | Stacked Auto Encoder for the P300 Component Detection | [Youtube](https://www.youtube.com/watch?v=ydpZaS1CCRg) |brain-computer interfaces|뇌-컴퓨터 인터페이스에서 P300 신호 검출을 쌓은 오토인코더로 수행, 정확도 상승.|
| Fundamental | A survey on Image Data Augmentation for DL | [Youtube](https://www.youtube.com/watch?v=TioeCk3yMCo&t=1073s) <br>[paper](https://link.springer.com/content/pdf/10.1186%2Fs40537-019-0197-0.pdf) | Data augmentation, 3D Human Pose Estimation|이미지 증강 기법을 종합적으로 정리, 3D 포즈 추정 등 다양한 태스크 성능 향상에 활용.|
| Fundamental | Training Confidence-calibrated classifiers for detecting out of distribution samples | [Youtube](https://www.youtube.com/watch?v=NOzDB2Rpbi0&t=150s) |classifiers|분류 모델이 OOD(분포외) 샘플을 탐지하도록 신뢰도 보정(Confidence Calibration) 훈련 기법.|
| Fundamental | AdamW | [Youtube](https://youtu.be/-Sd_zH_LHBo) <br> [blog](https://hiddenbeginner.github.io/deeplearning/paperreview/2019/12/29/paper_review_AdamW.html) |stochastic optimization|Adam + Weight Decay 분리로 일반화 성능 향상을 꾀한 최적화 기법.|
| Fundamental | Stargan | [Youtube](https://youtu.be/KO_mOGKdxOw) |Image-to-Image Translation, Translation|다중 도메인 간 이미지 변환을 하나의 모델로 통합, 얼굴 표정·헤어스타일 등 변경 가능.|
| Fundamental | Drop-out | [Youtube](https://www.youtube.com/watch?v=hOgQDK2-lA8) |Dropout, Data augmentation|뉴런 일부를 무작위로 비활성화해 과적합 억제, 일반화 성능 향상.|
| Fundamental | BLEU: a Method for Automatic Evaluation of Machine Translation | [Youtube](https://youtu.be/61my462pZk0) | evaluation|기계 번역 결과 품질을 n-그램 중복도로 측정하는 BLEU 지표 제안, 이후 표준 평가척도로 활용.|
| Fundamental | t-SNE| [Youtube](https://youtu.be/zCYKD3YfcSM) |Dimensionality Reduction|고차원 데이터(임베딩 등)를 2D·3D로 시각화해 구조 파악에 유용.|
| Fundamental | Gpipe| [Youtube](https://youtu.be/bbb0bLR0Faw) | Synchronous Pipeline Parallel, Model Parallel Methods|대형 모델 병렬화를 파이프라인 단위로 분할해 메모리 사용 효율 및 학습 속도 향상.|
| Fundamental | explainable ai| [Youtube](https://youtu.be/1WeLdfhRocI) |interpretability, sensitivity analysis |모델의 결정 과정을 해석·민감도 분석해 신뢰도·투명성 향상.|
| Fundamental | TAPAS| [Youtube](https://youtu.be/V2hPGrPqR0U) | Deep Tabular Learning, Table Question Answering Models|테이블 QA에 특화된 변형 BERT 모델로, 행·열 정보를 활용해 정교한 질의응답 수행.|
| Fundamental | Learning both Weights and Connections for Efficient Neural Networks| [Youtube](https://youtu.be/Gt2gvhcsPD8) |Neural networks |가중치와 연결(pruning)을 동시에 학습, 신경망 경량화와 정확도 유지 추구.|
| Fundamental | ReVACNN| [Youtube](https://youtu.be/EBaMig0nMoI) | Visualization, Convolutional Neural Network|CNN 내부 특징 맵을 시각화·분석하는 기법으로 모델 해석력을 높임.|
| Fundamental | The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks| [Youtube](https://youtu.be/EBaMig0nMoI) |Network Pruning |일부 가중치만 남겨도 초기화 시드가 동일하면 원래 수준 성능 달성 가능, 소위 ‘승리 티켓’ 가설 제시.|
| Fundamental | ALPHAGO : Mastering the game of Go with Deep Neural Networks and Tree Search| [Youtube](https://youtu.be/huMitou6zDs) |deep neural networks, tree search|정책망·가치망+몬테카를로 트리 탐색으로 바둑에서 프로기사 수준 초월, AI 역사적 성과.|
| Fundamental | A Baseline for Few-Shot Image Classification| [Youtube](https://youtu.be/qpCpo7wATto) |FEW-SHOT IMAGE CLASSIFICATION|소량 샘플로 분류하는 경우에 대한 베이스라인 접근을 정립, 후속 연구 지표로 활용.|
| Fundamental |  Sharp Minima Can Generalize For Deep Nets| [Youtube](https://youtu.be/5E9SFe5WU1s) |deep learning architectures|날카로운(Sharp) 최소점이 실제 일반화에 불리하다는 통념에 반대되는 실험 결과 제시.|
| Fundamental |  Pediatric Sleep Stage Classification  Using Multi-Domain Hybrid Neural Networks | [Youtube](https://youtu.be/mp72ClQT40s) |Multi-Domain Hybrid Neural Networks|여러 도메인(EEG 등) 특징을 융합한 하이브리드 모델로 소아 수면 단계 분류 성능 향상.|
| Fundamental |  Pruning from Scratch| [Youtube](https://youtu.be/ZBAhBHbXg40) |Network Pruning|처음부터 가지치기 구조로 학습을 시도, 기존 프루닝+재학습 절차 대비 효율적 가능성 탐색.|
| Fundamental |  Do We Need Zero Training Loss After Achieving Zero Training Error?| [Youtube](https://youtu.be/7HkFFFJar_E) |Training Loss |학습 오류가 0이 된 뒤에도 로스를 계속 낮출 필요가 있는지, 모델 일반화 관점에서 분석.|
| Fundamental |  Deep Recurrent Q-Learning for Partially Observable MDPs| [Youtube](https://youtu.be/M6hjaQSXEcE) | Atari Games, OpenAI Gym|부분 관측 MDP 환경에서 RNN+DQN으로 에이전트가 향상된 정책 학습 가능.|
| Fundamental |  Large Margin Deep Networks for Classification| [Youtube](https://youtu.be/Wl4kex_ZLqo) | Classification, Data Augmentation|마진 극대화 개념을 심층 신경망에 적용, 이미지 분류 등에서 일반화 성능 향상.|
| Fundamental | generating wikipedia by summarizing long sequences| [Youtube](https://youtu.be/C2xr5IA-4CM) | Document Summarization, Extractive Summarization|긴 시퀀스를 요약해 위키백과 페이지를 자동으로 생성·편집하는 접근.|
| Fundamental |  Plug and Play Language Models: A Simple Approach to Controlled Text Generation| [Youtube](https://youtu.be/QnsCHRCWEP4) |Language Modelling, Text Generation|사전학습된 LM에 제약(컨트롤) 모듈만 추가해 원하는 스타일·주제의 텍스트 생성 가능.|
| Fundamental | What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?|[Youtube](https://youtu.be/d7y42HfE6uI) | Depth Estimation, Semantic Segmentation|컴퓨터 비전에서 필요한 불확실성(에피스테믹·알레이터릭) 유형을 구분해 모델 신뢰성 높임.|
| Fundamental | KRED |[Youtube](https://youtu.be/Xq_FmQ-Sy1U) | Entity Embeddings, News Recommendation|뉴스 추천에서 개체(entity) 임베딩을 도입해 사용자 맞춤형 기사 배치를 정교화.|
| Fundamental | DIFUSCO: Graph based Diffusion Solvers for Combinatorial Optimization | [Youtube](https://youtu.be/ThcS_wFxTOY)|Diffusion, solver|그래프 기반 확산(diffusion) 기법으로 조합 최적화 문제(경로찾기 등)를 효과적으로 푸는 접근.|
| Fundamental | Early Stopping as nonparametric Variational  |[Youtube](https://youtu.be/q5AxUQr9KBg) |Variational Inference|조기 종료(Early Stopping)를 비모수적 변분 추론 관점으로 해석, 과적합 방지 설명 제공.|
| Fundamental | Sharpness Aware Minimization for efficeintly improving generalization   |[Youtube](https://youtu.be/iC3Y85W5tmM) |GENERALIZATION|손실 지형의 샤프니스(Sharpness)를 낮춰 일반화 성능을 높이는 SAM(Sharpness-Aware Minimization).|
| Fundamental | Neural Graph Collaborative Filtering   |[Youtube](https://youtu.be/ce0LrvVblCU) | Collaborative Filtering, Link Prediction|그래프 기반 협업필터링 모델로, 사용자-아이템 링크를 예측·추천.|
| Fundamental | Restricting the Flow: Information Bottlenecks for Attribution   |[Youtube](https://youtu.be/eUuXgkzR9MQ) | Decision Making|정보 병목(Bottleneck)을 활용해 모델 결정 경로를 추적·공헌도 측정(Attribution).|
| Fundamental | Real world Anomaly Detection in Surveillance Videos   |[Youtube](https://youtu.be/DYnvX5RaUL0) |Activity Recognition, Anomaly Detection In Surveillance Videos|실제 감시 영상에서의 이상 행동 감지를 위한 효율적인 딥러닝 접근.|
| Fundamental |  BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction   |[Youtube](https://youtu.be/aT0Fv1PzyV8) | Image Classification    Object Detection|블록 단위 재구성을 통한 정밀한 후처리 양자화 기법, 분류·검출 모두 높은 성능 유지.|
| Fundamental | Deep sets (2017 NIPS)   |[Youtube](https://youtu.be/EIZ3z823JQU) |designing models|집합 입력을 순서와 무관하게 처리하는 딥셋(DeepSets) 아이디어 제시, 대칭 함수 기반 학습.|
| Fundamental | StyleGAN2   |[Youtube](https://youtu.be/XMZWeqx5Vgg) |Generative Adversarial Networks, Generative Models|업그레이드된 아키텍처로, 더욱 사실적이고 다채로운 이미지를 생성할 수 있는 GAN 모델.|
| Fundamental | Beyond Synthetic Noise: Deep Learning on Controlled Noisy Labels  |[Youtube](https://youtu.be/s83wjMtdQh8) | Image Classification|인위적으로 제어된 노이즈 라벨 환경에서의 딥러닝 성능 분석, 노이즈 강인성 연구.|
| Fundamental | Deep Reinforcement Learning for Online Advertising Impression in Recommender Systems   |[Youtube](https://youtu.be/kTT7YsHWodU) |Recommendation Systems|온라인 광고 노출(Impression)을 강화학습으로 최적화, 추천 시스템 수익 극대화 시도.|
| Fundamental | Longformer: The Long-Document Transformer   |[Youtube](https://youtu.be/i7aiBMDExmA) |Language Modelling, Question Answering|희소 어텐션(sparse attention)으로 긴 문서(최대 수천~수만 토큰) 처리 가능, QA·분류 성능 향상.|
| Fundamental | soft actor critic   |[Youtube](https://youtu.be/HK7Y20Bt7qM) |Policy Gradient Methods|액터-크리틱에 엔트로피 보상을 추가해 탐색 다양성·학습 안정성을 높이는 강화학습 기법.|
| Fundamental | Loss Function Discovery for Object Detection via Convergence-Simulation Driven Search   |[Youtube](https://youtu.be/P_yXwbPefQ8) |Object Detection, Loss Function Search	|시뮬레이션 기반 탐색으로 최적의 손실 함수를 자동으로 찾는 기법을 객체 검출에 적용.|
| Fundamental |The Deep Bootstrap Framework: Good Online Learners are Good Offline Generalizers |[Youtube](https://youtu.be/WwXzLCmWvqM) |Online Learning, Bootstrap Framework|	온라인에서 잘 학습하는 부트스트랩 기법이 오프라인 일반화에도 유리함을 보임.|
| Fundamental | Meta HIN  |[Youtube](https://youtu.be/v8bma8QMK7k) |Heterogeneous Info Networks, Graph Representation|	이기종 정보 네트워크(heterogeneous graph) 상에서 메타 러닝 기법을 적용해 구조적 표현을 학습.|
| Fundamental | When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations  |[Youtube](https://youtu.be/gB_YmwC1AY0) | Domain Generalization, Fine-Grained Classification|	별도 사전학습이나 강력한 증강 없이도 비전 트랜스포머가 레스넷보다 우수한 결과를 보이는 케이스 분석.|
| Fundamental | Self similarity Student for Partial Label Histopathology Image Segmentation  |[Youtube](https://youtu.be/2Kw-xgpHTqY) |Partial-Label, Histopathology Segmentation|	부분 라벨(Partial Label) 환경의 조직 이미지(병리) 세그멘테이션에서 자가 유사도 방식 활용.|
| Fundamental | ANALYSING MATHEMATICAL REASONING ABILITIES OF NEURAL MODELS  |[Youtube](https://youtu.be/jE1gJQH5OJI) |Math Reasoning, Word Problem Solving|	신경망의 수학 추론(문제 풀이) 능력을 평가하고 한계를 분석한 연구.|
| Fundamental | Self-training Improves Pre-training for Natural Language Understanding |[Youtube](https://youtu.be/9iJLzmrUN-8) | Self-training, Few-Shot Learning|	셀프 트레이닝을 통해 NLU 사전학습 모델의 적은 데이터 환경 성능을 향상.|
| Fundamental | Preference Amplification in Recommender Systems |[Youtube](https://youtu.be/dM3kDjpSzBk) | Recommender Systems, Preference Amplification	|추천 시스템에서 사용자 취향이 극단적으로 증폭되는 현상을 분석·완화하는 연구.|
| Fundamental | Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation |[Youtube](https://youtu.be/UO5iqW3iTFU) | Source Hypothesis Transfer, Unsupervised Domain Adaptation	|소스 데이터를 직접 접근하지 않고도 가설(모델 파라미터)만으로 도메인 적응을 수행하는 기법 제안.|
| Fundamental | Learning to Learn at Test Time RNNs | [Youtube](https://youtu.be/D0nyybmIRSM)|Meta-Learning, Test-Time Adaptation|	테스트 단계에서도 순환신경망(RNN)이 자체 학습(메타 러닝)을 수행해 예측 성능 개선.|
| Fundamental | Double Gumbel Q-Learning | [Youtube](https://youtu.be/i1FEbw0kmKM)|RL, Double Q-Learning, Gumbel Noise|	이중 Q-Learning에 Gumbel 분포 노이즈를 사용해 탐색성과 안정성을 높인 강화학습 기법.|
| Fundamental | Confidence Estimation Using Unlabeled Data | [Youtube](https://youtu.be/sgUNQwADiCw)|Confidence Estimation, Semi-supervised Learning|	라벨이 없는 데이터를 활용해 분류 모델의 신뢰도(Confidence)를 추정·보정.|
| Fundamental | Evaluating Classifiers by Means of Test Data with Noisy Labels |[Youtube](https://youtu.be/xByR5oix9ms) |Classifier Evaluation, Noisy Labels	|라벨 노이즈가 있는 테스트 데이터에서 분류기를 평가하는 방법 제안.|
| Fundamental | Progressive Identification of True Labels for Partial-Label Learning |[Youtube](https://youtu.be/QsvgzjfSFhg) | PPartial Label Learning, Progressive Label Refinement	|부분 라벨 환경에서 점진적으로 진짜 라벨을 식별해 학습 정밀도를 높이는 접근.|
| Fundamental | Fine-grained Interest Matching For Neural News Recommendation |[Youtube](https://youtu.be/XW93QvbFlaQ) |News Recommendation, Fine-grained Matching	|뉴스 추천에서 세분화된 사용자 관심사(interest) 매칭을 통해 추천 품질 향상.|
| Fundamental | Adversarial Reinforced Learning for Unsupervised Domain Adaptation |[Youtube](https://youtu.be/zeBMMXKj39U) |Transfer Learning, RL-based Domain Adaptation|	강화학습과 적대적(Adversarial) 방식을 결합해 도메인 적응 시 라벨 없는 데이터 활용도 개선.|
| Fundamental | Neural Tangent Kernel - Convergence and Generalization in neural Network |[Youtube](https://youtu.be/vDQWwOqQ7mo) |Neural Tangent Kernel, Gaussian Process	NTK| 관점에서 신경망의 수렴·일반화 특성을 분석, 가우시안 프로세스와의 유사점 탐구.|
| Fundamental | Intriguing Properties of Contrastive Losses |[Youtube](https://youtu.be/uzsI-dEoK2c) |Contrastive Learning, Data Augmentation|	대조학습 손실 함수들의 숨은 특성·데이터 증강 전략이 모델 임베딩에 미치는 영향 탐색.|
| Fundamental | Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets |[Youtube](https://youtu.be/mcnSN645xUE) |Algorithmic Generalization, Overfitting|	작은 알고리즘 데이터셋에서 한참 오버피팅 후 갑작스런 일반화(grokking) 현상을 분석한 연구.|
| Fundamental | Transformer Interpretability Beyond Attention Visualization |[Youtube](https://youtu.be/XCED5bd2WT0) | Transformer Explanation, Interpretation	|트랜스포머 해석 시 어텐션 시각화만으로는 부족, 추가 기법으로 내부 작동 원리 파악 시도.|
| Fundamental | How does unlabeled data improve generalization in self-training? |[Youtube](https://youtu.be/t7dY-k-JBPA) |Self-training, Semi-supervised Learning|	라벨 없는 데이터가 셀프 트레이닝에서 일반화 성능 향상에 기여하는 원리를 분석.|
| Fundamental | Rainbow: Combining Improvements in Deep Reinforcement Learning |[Youtube](https://youtu.be/oC1AOIefjT8) |Rainbow DQN, Montezuma's Revenge	DQN| 계열 기법(우선순위 리플레이·NoisyNet 등)을 결합한 Rainbow가 고난이도 게임을 잘 해결.|
| Fundamental | Once-for-All: Train One Network and Specialize it for Efficient Deployment |[Youtube](https://youtu.be/xxvvVUhG5ik) |Neural Architecture Search, Model Specialization|	단 한 번의 거대 네트워크 학습 후, 기기별 요구사항에 맞춰 하위 네트워크를 추출·사용.|
| Fundamental | Effective Diversity in Population Based Reinforce Learning |[Youtube](https://youtu.be/AG2uYbbXZuo) |Population-based RL, Diversity Optimization|	개체군(인구) 기반 RL에서 다양성 유지를 통해 탐색 성능과 수렴성을 향상.|
| Fundamental | Fine-Tuning can Distort Pretrained Features and Under perform Out-of-Distribution |[Youtube](https://youtu.be/okBhSW2x2eQ) |Fine-tuning Instability, OOD Generalization|	파인튜닝 과정이 사전학습된 특징을 망가뜨려 OOD 일반화 성능이 떨어지는 문제 고찰.|
| Fundamental | GCN-LRP explanationexploring latent attention of graph convolutional networks |[Youtube](https://youtu.be/u_sy3UfFJsU) |Graph Interpretability, GCN Explanation|	그래프 합성곱 네트워크(GCN)의 예측 근거를 LRP(Layer-wise Relevance Propagation) 방식으로 해석.|
| Fundamental | GQA : Training Generalized Multi Query Transformer Models from Multi Head Checkpoint | [Youtube](https://youtu.be/lvSDQujIN-c)|Multi-Query Transformer, Checkpoint Merging|	멀티 헤드 체크포인트를 합쳐 다목적 질의(Q) 트랜스포머 모델로 훈련.|
| Fundamental | Towards Safe Online Reinforced Learning in Computer Systems |[Youtube](https://youtu.be/LQRisuX0Ppc) |Safe Online RL, Computer Systems|	실제 컴퓨터 시스템 환경에서 안전(safety) 제약을 고려한 온라인 RL 연구.|
| Fundamental | Conflict-Averse Gradient Descent for Multi-task Learning |[Youtube](https://youtu.be/Q9E79bTBkQQ) |Multi-task Learning, Gradient-based Optimization|	여러 태스크 간 충돌을 줄이는 경사하강 기법으로 멀티태스크 학습 효율 향상.|
| Fundamental | Explainability Methods for Graph Convolutional Neural Networks |[Youtube](https://youtu.be/DJfbq_Ifnj8) |Graph Explainability, GCN|	GCN 모델을 해석하기 위한 다양한 방법론(서브그래프 추출 등) 정리.|
| Fundamental | Bayesian Action Decoder for Deep Multi-Agent Reinforcement Learning |[Youtube](https://youtu.be/P_Ca8zTSkM8) |Multi-Agent RL, Bayesian Inference|	다중 에이전트 강화학습에서 액션 디코더를 베이지안 방식으로 설계해 불확실성·상호작용 처리.|
| Fundamental | Efficiently Identifying Task Grouping for Multi-Task Learning |[Youtube](https://youtu.be/nqHMYewBPXg) |Multi-Task Learning, Task Grouping	여러 태스크를 효과적으로 묶어 학습 비용·성능을 최적화하는 방법론.
| Fundamental | DeepFM: A Factorization-Machine based Neural Network for CTR Prediction |[Youtube](https://youtu.be/otjvQYbGzEI) |CTR Prediction, Recommendation|	팩터라이제이션 머신(FM)과 딥러닝을 결합해 클릭확률 예측(CTR) 정확도 향상.|
| Fundamental | Deep Neural Networks for YouTube Recommendations |[Youtube](https://youtu.be/kk5nXrhUs0Q) |YouTube Recommendation, Ranking|	유튜브 대규모 추천 시스템에 특화된 딥러닝 구조로 실시간 사용자 맞춤 추천 구현.|
| Fundamental | Genome wide prediction of disease variant effects with a deep protein lang | [Youtube](https://youtu.be/U26AvD0aCkQ)|Protein LM, Disease Variant Prediction|	단백질 언어모델을 활용해 질병 관련 유전자 변이의 영향을 전장(Genome-wide) 예측.|
| reinforcement learning | Reinforcement Learning for Optimizing RAG for Domain Chatbots | [Youtube](https://youtu.be/J4Z2BA_5GnQ)|RAG Optimization, Chatbot RL|	도메인 챗봇에서 RAG(Retrieval-Augmented Generation)을 강화학습으로 최적화.|
| reinforcement learning | Deep reinforcement learning from human preferences |[Youtube](https://youtu.be/B48rm3jeGmQ) |Human-in-the-Loop RL, Preference Learning|	인간의 선호도 신호를 이용해 RL 에이전트를 학습, 보상 함수 추정 없이 정책 최적화.|
| reinforcement learning  | Distributional Reinforcement Learning via Moment Matching |[Youtube](https://youtu.be/rYXOGP52AA0) |Distributional RL, Moment Matching	Q|함수를 분포로 다루는 Distributional RL 기법에서 모멘트 매칭으로 학습 안정화.|
| reinforcement learning  | Online Continual Learning on class Incremental Blurry Task Configuration with anytime inference |[Youtube](https://youtu.be/YjjWoMYGwPI) |Continual RL, Class-Incremental|	순차적·증분적 태스크 환경에서 RL 에이전트가 지속적으로 학습·추론하는 방법 제안.|
| reinforcement learning | Dream To Control : Learning Behaviors by Latent Imagination |[Youtube](https://youtu.be/Bbduj2KyQ2Y) |Model-Based RL, Dreamer Approach|	잠재공간에서 ‘꿈(rollout)’을 만들어 행동 정책을 학습하는 Dreamer 방식 확장.|
| reinforcement learning  | Proximal Policy Optimization Algorithms |[Youtube](https://youtu.be/kk5nXrhUs0Q) |Policy Gradient, PPO	KL| 발산을 이용해 정책 갱신 폭을 제한, 샘플 효율성과 안정성을 높인 RL 기법.|
| reinforcement learning | Reinforced Genetic Algorithm Learning for Optimizing Computation Graphs |[Youtube](https://youtu.be/fs4J2Oq0nkg) |Genetic Algorithm, RL for Computation Graph	GA와 RL을 결합해 연산 그래프 최적화(컴파일러 최적화 등)에 적용.
| reinforcement learning | VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning |[Youtube](https://youtu.be/mE4uZWmktV4) |Bayes-Adaptive RL, Meta-Learning|	환경 불확실성을 베이지안으로 다루고, 메타학습을 통해 빠른 적응을 구현하는 RL 기법.|
| reinforcement learning | mPLUG: Effective and Efficient Vision-Language Learning byCross-modal Skip-connections |[Youtube](https://youtu.be/6uECXx75jIo) |Vision-Language Model, Cross-Modal Learning|	스킵 연결로 멀티모달(이미지+텍스트) 학습 효율을 높이는 mPLUG 아키텍처. (RL과 직접 관련은 미미하지만, 섹션상 RL로 분류됨)|
| reinforcement learning | MOReL: Model-Based OfflineReinforcement Learning |[Youtube](https://youtu.be/mueizVwVqek) |Offline RL, Model-Based|	실제 환경 데이터 없이 모델(환경 시뮬레이션)로 학습하는 오프라인 강화학습 기법.|
| reinforcement learning | RL upside down |[Youtube](https://youtu.be/bsBvKdKCc1E) |Reverse RL, Goal-Conditioned|	보상 대신 목표 상태부터 행동을 추론하는 역발상 RL 접근, 목표 지향 문제에 적용.|
| reinforcement learning | Trans Dreamer |[Youtube](https://youtu.be/TqCqKROQk6M) |Transformer-based Dreamer, Model-Based RL	Dreamer| 모델을 트랜스포머 구조로 확장해 장기 시퀀스 행동 계획 정확도 개선.|
| reinforcement learning | Direct Preference Optimization Your Language Model is Secretly a Reward Model | [Youtube](https://youtu.be/uX7_wXZQ_ZY)|LLM as Reward Model, Preference Optimization|	언어모델(LLM)을 보상 모델로 간주해 직접 선호 최적화(Direct Preference Optimization) 수행.
| reinforcement learning | Trajectory Transformer |[Youtube](https://youtu.be/sqbxOkDPz4k) |Decision Transformer, Sequence Modeling|	행동 궤적(trajectory)을 트랜스포머로 모델링해 RL 문제를 시퀀스 예측으로 접근.|
| reinforcement learning | Addressing Optimism Bias in Sequence Modeling for RL |[Youtube](https://youtu.be/yBAtyOqF378) |Sequence Modeling RL, Optimism Bias|	시퀀스 모델 기반 RL에서 과도한 낙관 편향(optimism bias)을 완화하는 방법 연구.
| reinforcement learning | Online Decision Transformer | [Youtube](https://youtu.be/6yAb6_nyn3A)|Online RL, Decision Transformer|	의사결정 트랜스포머(Decision Transformer)를 온라인 학습 시나리오로 확장한 기법.
| reinforcement learning | RLPROMP: Optimizing Discrete Text Prompts with Reinforcement Learning | [Youtube](https://youtu.be/L8dsadlYc18)|Prompt Optimization, RL for NLP	NLP| 프롬프트를 RL로 최적화해, 원하는 응답·출력 양식을 유도하는 기법 (Discrete Prompt).|
