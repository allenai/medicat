{
    "dataset_reader": {
        "type": "docfigure",
        // "image_root": "/net/nfs.corp/allennlp/data/docfigure/images/",
        "image_root": "/images/"
    },
    // "train_data_path": "/net/nfs.corp/allennlp/data/docfigure/annotation/train.txt",
    // "validation_data_path": "/net/nfs.corp/allennlp/data/docfigure/annotation/test.txt",
    "train_data_path": "/annotations/train.txt",
    "validation_data_path": "/annotations/test.txt",
    "model": {
        "type": "image_classifier",
        "backbone": "resnet101",
        "dropout_prob": 0.1,
    },
    "iterator": {
      "type": "basic",
      "batch_size": 64
    },
    "trainer": {
      "num_epochs": 50,
      "cuda_device": 0,
      "num_serialized_models_to_keep": 1,
      "optimizer": {
        "type": "adam",
        "lr": 0.001,
      }
    }
}
