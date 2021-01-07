{
        "dataset_reader": {
		"type": "object",
                "image_root": "/images",
		"pass_boxes": true,
	},
	"validation_dataset_reader": {
		"type": "object",
		"image_root": "/images",
		"pass_boxes": true,
	},
	"train_data_path": "/data/train.jsonl",
	"validation_data_path": "/data/valid.jsonl",
        "model": {
		"type": "object",
	},
	"iterator": {
		"type": "basic",
		"batch_size": 10
	},
	"trainer": {
		"num_epochs": 50,
		"cuda_device": 0,
		"patience": 5,
		"num_serialized_models_to_keep": 1,
		"optimizer": {
			"type": "adam",
			"lr": std.extVar("trainer_lr"),
		},
		"validation_metric": "+mAP"
	}
}
