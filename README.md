# improved_keras_english_resume_IE


This code is the paper with:
@article{甘程光2021英文履歴書データ抽出システムへの,
  title={英文履歴書データ抽出システムへの BERT 適用性の検討},
  author={甘程光 and 高橋良英 and others},
  journal={2021 年度 情報処理学会関西支部 支部大会 講演論文集},
  volume={2021},
  year={2021}
}

The dataset is already publicly available at kaggle. You can download it at this URL below.
https://www.kaggle.com/datasets/chingkuangkam/resume-text-classification-dataset

Put the dataset into this folder (demo\data\training_data) and you can use it.


Bi-LSTM-Attention and CNN-Attention Model:

The main code are base in chen0040/keras-english-resume-parser-and-analyzer for improvement.
Add the Attention layer in last original output layer.



BERT Model:

BERT Model is base in sujianlin's bert4keras.

You need pip install bert4keras.
