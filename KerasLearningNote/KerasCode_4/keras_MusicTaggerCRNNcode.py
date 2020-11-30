from keras.applications.music_tagger_crnn import MusicTaggerCRNN
from keras.applications.music_tagger_crnn import preprocess_input, decode_predictions
import numpy as np
 
# 1. Tagging
model = MusicTaggerCRNN(weights='msd')
 
audio_path = 'audio_file.mp3'
melgram = preprocess_input(audio_path)
melgrams = np.expand_dims(melgram, axis=0)
 
preds = model.predict(melgrams)
print('Predicted:')
print(decode_predictions(preds))
# print: ('Predicted:', [[('rock', 0.097071797), ('pop', 0.042456303), ('alternative', 0.032439161), ('indie', 0.024491295), ('female vocalists', 0.016455274)]])
 
#. 2. Feature extraction
model = MusicTaggerCRNN(weights='msd', include_top=False)
 
audio_path = 'audio_file.mp3'
melgram = preprocess_input(audio_path)
melgrams = np.expand_dims(melgram, axis=0)
 
feats = model.predict(melgrams)
print('Features:')
print(feats[0, :10])
# print: ('Features:', [-0.19160545 0.94259131 -0.9991011 0.47644514 -0.19089699 0.99033844 0.1103896 -0.00340496 0.14823607 0.59856361])
