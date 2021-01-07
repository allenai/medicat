local scibert_path = "/net/nfs.corp/allennlp/sanjays/scibert/scibert_scivocab_uncased";
local num_epochs = 100;
local batch_size = 16;
{
    "dataset_reader": {
        "type": "image_text_matching",
        "image_root": "/net/nfs.corp/allennlp/sanjays/roco_train_images",
        "scibert_path": scibert_path,
        "max_sequence_length": 256,
        "limit": 100,
    },
    "validation_dataset_reader": {
        "type": "image_text_matching",
        "image_root": "/net/nfs.corp/allennlp/sanjays/roco_val_images",
        "scibert_path": scibert_path,
        "limit": 100,
    },
    // "train_data_path": ["/net/nfs.corp/allennlp/sanjays/roco_train.jsonl"],
    "train_data_path": ["/net/nfs.corp/allennlp/sanjays/roco_train_with_refs.jsonl"],
    "validation_data_path": ["/net/nfs.corp/allennlp/sanjays/roco_validation.jsonl"],
    "model": {
        "type": "image_text_matching",
        "pretrained": true,
        "pretrained_bert": true,
        "scibert_path": scibert_path,
        "fusion_layer": 6,
        "dropout": 0.1,
    },
    "iterator": {
      "type": "basic",
      "batch_size": batch_size
    },
    "trainer": {
      "type": "image_text_matching",
      "retrieve_text": false,
      // "patience": 5,
      "num_epochs": num_epochs,
      "cuda_device": 0,
      "num_serialized_models_to_keep": 1,
      "optimizer": {
        "type": "adam",
        "lr": 1e-5,
      },
      "validation_metric": "+accuracy",
    },
    // "random_seed": std.extVar("random seed - optional"),
    // "numpy_seed": std.extVar("random seed - optional"),
    // "pytorch_seed": std.extVar("random seed - optional"),
}
