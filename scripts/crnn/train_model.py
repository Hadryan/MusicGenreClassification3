from IModel import Model
from music_tagger_cnn import MusicTaggerCNN
from tagger_net import MusicTaggerCRNN

model = Model(MusicTaggerCRNN, None)
model.fit()

