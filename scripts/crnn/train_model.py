from IModel import Model
from genre_rec_CNN import GenreRecCNN
from genre_rec_CRNN import GenreRecCRNN

model = Model(GenreRecCNN)
model.fit()