import string
from PIL import Image
import os
from pickle import dump, load
import numpy as np
from keras.applications.xception import Xception
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.merge import add
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tqdm import tqdm_notebook as tqdm

# Load text data from special file path
def get_textData(file_path):
    # Opening the file as read only
    textFile = open(file_path, 'r')
    data = textFile.read()
    textFile.close()
    return data
# Loading images and image recognition text data
def get_annotation(ann_filePath):
    ann_data = get_textData(ann_filePath)
    recog_textArray = ann_data.split('\n')
    recog_text ={}
    for recog_item in recog_textArray[:-1]:
        img_name, item_text = recog_item.split('\t')
        if img_name[:-2] not in recog_text:
            recog_text[img_name[:-2]] = [item_text]
        else:
            recog_text[img_name[:-2]].append(item_text)
    return recog_text
# Pre-processing for Data cleaning
def Pre_process(recog_text):
    string_T = str.maketrans('','',string.punctuation)
    for img_name, item_data in recog_text.items():
        for i, img_text in enumerate(item_data):
            img_text.replace("-"," ")
            d_text = img_text.split()
            # Get lowercase text
            d_text = [word.lower() for word in d_text]
            # Get token and remove punctuation.
            d_text = [word.translate(string_T) for word in d_text]
            d_text = [word for word in d_text if(len(word)>1)]
            #Remove tokens with numbers
            d_text = [word for word in d_text if(word.isalpha())]
            # Get string
            img_text = ' '.join(d_text)
            recog_text[img_name][i]= img_text
    return recog_text
# Get vocabulary of words
def get_vocab(cleaned_desc):
    v_data = set()
    for key in cleaned_desc.keys():
        [v_data.update(d.split()) for d in cleaned_desc[key]]
    return v_data
# Save all recognition text data in one file
def save_ann(cleaned_desc, file_path):
    lines = list()
    for key, desc_list in cleaned_desc.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc )
    data = "\n".join(lines)
    file = open(file_path,"w")
    file.write(data)
    file.close()
# Get features from each image
def get_features(imge_folder):
    model = Xception(include_top=False, pooling='avg')
    features = {}
    for img_name in tqdm(os.listdir(imge_folder)):
        image_fullpath = imge_folder + "/" + img_name
        image = Image.open(image_fullpath)
        image = image.resize((299, 299))
        image = np.expand_dims(image, axis=0)
        image = image / 127.5
        image = image - 1.0
        feature = model.predict(image)
        features[img_name] = feature
    return features
# Get image name array
def get_images(folder):
    file = get_textData(folder)
    images = file.split("\n")[:-1]
    return images
# Get cleaned text description data from saved file
def Get_clean_textData(folder, images):

    file = get_textData(folder)
    ann_texts = {}
    for line in file.split("\n"):
        words = line.split()
        if len(words) < 1:
            continue
        image, image_text = words[0], words[1:]
        if image in images:
            if image not in ann_texts:
                ann_texts[image] = []
            desc = '<start> ' + " ".join(image_text) + ' <end>'
            ann_texts[image].append(desc)
    return ann_texts
# Load training features
def load_features(images):
    all_features = load(open("features.p","rb"))
    features = {k:all_features[k] for k in images}
    return features
# Get description array from dict data
def get_senList(dict_data):
    sen_list = []
    for key in dict_data.keys():
        [sen_list.append(d) for d in dict_data[key]]
    return sen_list
# Create token
def create_token(dict_data):
    sen_list = get_senList(dict_data)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sen_list)
    return tokenizer
# Get maximum length of annotation text string
def max_length(ann_text):
    sen_list = get_senList(ann_text)
    return max(len(d.split()) for d in sen_list)

# Create input and output pair for training according to each image
def create_sequences(tokenizer, max_length, sen_list, feature, vocab_size):
    data1, data2, y_ann = list(), list(), list()
    # Get each text annotation for each image
    for ann_t in sen_list:
        seq = tokenizer.texts_to_sequences([ann_t])[0]
        # split one sentence
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # get input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # Get output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            data1.append(feature)
            data2.append(in_seq)
            y_ann.append(out_seq)
    return np.array(data1), np.array(data2), np.array(y_ann)
def data_generator(ann_texts, features, tokenizer, max_length, vocab_size):
    while 1:
        for key, description_list in ann_texts.items():
            #retrieve photo features
            feature = features[key][0]
            img_input, seq_in, word_out = create_sequences(tokenizer, max_length, description_list, feature, vocab_size)
            yield [[img_input, seq_in], word_out]
def LSTM_model(vocab_size, max_length):
    # features from the CNN model squeezed from 2048 to 256 nodes
    in_layer1 = Input(shape=(2048,))
    hidden1 = Dropout(0.5)(in_layer1)
    hidden2 = Dense(256, activation='relu')(hidden1)
    # LSTM sequence model
    in_layer2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(in_layer2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # Merging both models
    decoder1 = add([hidden2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[in_layer1, in_layer2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model
# Annotation text file and image data set
text_annFile_path = "Flickr8k_text"
Image_file_path = "Flicker8k_Dataset"
# Token file
filename = text_annFile_path + "/" + "Flickr8k.token.txt"
# Get annotation text dataw
recog_textData = get_annotation(filename)
# Cleaning the recog_textData
clean_recog_textData = Pre_process(recog_textData)
#Create vocabulary
vocab_data = get_vocab(clean_recog_textData)
# Save cleaned recog_textData
save_ann(clean_recog_textData, "descriptions.txt")
# Get feature sets from image directory and save them
features = get_features(Image_file_path)
dump(features, open("features.p","wb"))
# Load feature sets from saved file
features = load(open("features.p","rb"))
# Get training data
train_file = text_annFile_path + "/" + "Flickr_8k.trainImages.txt"
train_images = get_images(train_file)
train_ann = Get_clean_textData("descriptions.txt", train_images)
train_features = load_features(train_images)
# Get tokenizer and save it
tokenizer = create_token(train_ann)
dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
# Get max length of annotation text
max_length = max_length(recog_textData)

model = LSTM_model(vocab_size, max_length)
epochs = 10
steps = len(train_ann)
# Create model path to save
os.mkdir("models")
for i in range(epochs):
    data_gen = data_generator(train_ann, train_features, tokenizer, max_length, vocab_size)
    model.fit_generator(data_gen, epochs=1, steps_per_epoch= steps, verbose=1)
    model.save("models/model_" + str(i) + ".h5")



