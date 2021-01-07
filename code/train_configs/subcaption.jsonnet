local bert_model = "allenai/scibert_scivocab_cased";
local do_lowercase = false;

{
  "dataset_reader": {
    "type": "subcaption_ner",
    // "type": "subcaption_box",
    "do_lowercase": do_lowercase,
    "token_indexers": {
      "bert": {
          "type": "pretrained_transformer2",
          "model_name": bert_model,
          "do_lowercase": do_lowercase,
          "namespace": "bert"
          // "use_starting_offsets": true
      }
    }
  },
  "train_data_path": "/data/train.jsonl",
  "validation_data_path": "/data/valid.jsonl",
  "model": {
    "type": "bert_crf_tagger",
    // "type": "bert_box_crf_tagger",
    "label_encoding": "BIOUL",
    "constrain_crf_decoding": true,
    "dropout": std.extVar("model_dropout"),
    "include_start_end_transitions": true,
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
            "bert": ["bert"]//, "bert-offsets"]
        },
        "token_embedders": {
            "bert": {
                "type": "pretrained_transformer2",
                "model_name": bert_model,
            }
        }
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 8,
    "cache_instances": true
   },
  "trainer": {
    "optimizer": {
        "type": "bert_adam",
        "lr": std.extVar("trainer_lr"),
        "weight_decay": std.extVar("trainer_decay"),
        "parameter_groups": [
          [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}]
        ]
    },
    "validation_metric": "+subcaption_f1",
    "num_serialized_models_to_keep": 1,
    "num_epochs": 50,
    "patience": 5,
    "should_log_learning_rate": true,
    "cuda_device": 0
  }
}
