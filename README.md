# Audio Classification for the Urban Sound Datasets

Steps to train a model:
* Download the [urban sound datasets](https://urbansounddataset.weebly.com/), and unzip it in the **raw_data** folder
<br>There should be 10 sub-folders in the raw_data folder
* Change the sample rate of the audio
<br>`python src/wav16000.py --raw_data_dir raw_data --data_16000_dir wav_16000`
* Perfrom STFT to the audio files and save each audio as a tensor
<br>`python src/audio_pth.py --data_16000_dir wav_16000 --data_dir data`
* Create train/eval/test manifest files
<br>`python src/manifest.py`
* Train
<br>`python src/train.py`
<br>The best model will be store in `./log/<date_time>/best.model.pth`
* TODO
Correct the train/eval/test splits [AVOID COMMON PITFALLS](https://urbansounddataset.weebly.com/urbansound8k.html)
