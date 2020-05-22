# Face Mask Detection (COVID-19)
An effective solution to prevent coronavirus from spreading.<br>
By Rajat Agrawal

<h2>Dependencies</h2> (Also mentioned in requirements.txt)
<ul>
  <li>Protobuf</li>
  <li>Python-tk</li>
  <li>Pillow 1.0</li>
  <li>lxml</li>
  <li>Jupyter notebook</li>
  <li>numpy</li>
  <li>Matplotlib</li>
  <li>Tensorflow (1.14.0 or greater)</li>
  <li>Cython</li>
  <li>contextlib2</li>
  <li>cocoapi</li>
</ul>

  Use <b>pip</b> to install any missing dependencies.
## Before you start -
Follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) for tensorflow object detection API.<br>
Now clone this repo into your tensorflow\models\research\object_detection folder.<br>
Steps to creat your own mask detector.<br>
<ol>
  <li>Gathering data</li>
  <li>Labeling data</li>
  <li>Generating TFRecords for training</li>
  <li>Configuring training</li>
  <li>Training model</li>
  <li>Exporting inference graph</li>
  <li>Testing face mask detector</li>
</ol>

## 1. Gathering data
Scrap pictures (people wearing and not wearing mask) from internet (Use google-images-download).<br>
`pip install google-images-download`<br>
Keep 80% of data in object_detection\images\train and 20% data in object_detection\images\test folder.

## 2. Labeling data
Label your data using [LabelImg](https://tzutalin.github.io/labelImg/).<br>
Move labelmap.pbtxt to training folder (find in repo).

## 3. Generating TFRecords for training
Run these commands in terminal(cmd).<br>
(find `xml_to_csv.py` and `generate_tfrecord.py` in repo).
```
python xml_to_csv.py
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
  ```
  
## 4. Configuring training
Doownload model of your choice (ex: faster_rcnn_inception_v2_coco) from [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).<br>
Now copy the config file (pipeline.config) to training folder.<br><br>

Edit the following lines in pipeline.config file -<br>
Change the number of classes to 2.<br><br>

Change `fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"`<br>
to `fine_tune_checkpoint: "<PATH TO>/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"`<br><br>

Under `train_input_reader`<br>
Change `input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record"`<br>
to `input_path: "<PATH TO>/object_detection/train.record"`<br>
and<br>
Change `label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"`<br>
to `label_map_path: "<PATH TO>/object_detection/training/labelmap.pbtxt"`<br><br>

Under `eval_config`<br>
Change num_examples to number of your test images.<br><br>

Under `eval_input_reader`<br>
Change `input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record"`<br>
to `input_path: "<PATH TO>/object_detection/test.record"`<br>
and<br>
Change `label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"`<br>
to `label_map_path: "<PATH TO>/object_detection/training/labelmap.pbtxt"`<br><br>

## 5. Training model
Run the command in terminal(cmd).<br>
```python model_main.py --logtostderr --model_dir=training/ --pipeline_config_path=training/pipeline.config --num_train_steps=5000 --NUM_EVAL_STEPS=500```

## 6. Exporting inference graph
Chnage `model.ckpt-XXXX` to the last saved model name in training folder.<br>
Run the command in terminal(cmd).<br>
```python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph```<br><br>

## 7. Testing face mask detector
Run `faceMaskDetection.py`
