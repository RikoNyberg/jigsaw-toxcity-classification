{

  iterator: {
    type: 'bucket',
    sorting_keys: [['tokens', 'num_tokens']],
    batch_size: 512
  },

  model: {
    type: 'toxcity_lstm'
    embedder: {
      tokens: {
        type: 'embedding',
        pretrained_file: "(http://nlp.stanford.edu/data/glove.840B.300d.zip)#glove.840B.300d.txt",
        embedding_dim: 300,
        trainable: false
      }
    },
    encoder: {
      type: 'lstm',
      input_size: 300,
      hidden_size: 128,
      bidirectional: true
    }
  },

  trainer: {
    num_epochs: 4,
    patience: 3,
    cuda_device: 0,
    grad_clipping: 5.0,
    validation_metric: '-loss',
    optimizer: {
      type: 'adam',
      lr: 0.001
    }
  },

  dataset_reader: {
    type: 'toxicity_data_reader',
    lazy: true
  },
  train_data_path: 'data/train.csv',
}
