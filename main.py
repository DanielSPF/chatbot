from nltk.stem import RSLPStemmer
import os
import pickle
import json
import random
import tensorflow
import tflearn
import numpy
import nltk
nltk.download('rslp')
stemmer = RSLPStemmer()
nltk.download('punkt')


# Abre o json
with open("intents.json") as file:
    data = json.load(file)

# Os dados para treino já foram convertidos?
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:  # Preenche os dados para treinar a rede
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # Preenche as listas
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern, language="portuguese")
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Padroniza as palavras
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    # Remove duplicatas
    words = sorted(list(set(words)))
    # Ordena as tags
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    # Converte as entradas e saídas desejadas em 0 e 1
    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # Preenche as entradas para treino e as saídas
    training = numpy.array(training)
    output = numpy.array(output)

    """
        Uma vez que as entradas e saídas foram convertidas elas são salvas para não precisar converter novamente.
    """
    # with open("data.pickle", "wb") as f:
    #pickle.dump((words, labels, training, output), f)

# Limpa a pilha de gráficos padrão e redefine o gráfico padrão global do tensorflow.
tensorflow.reset_default_graph()

# Define a estrutura da rede neural
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# Define o modelo da rede como uma Deep Neural Network
model = tflearn.DNN(net)

# O modelo já existe?
if os.path.exists("model.tflearn.meta"):
    model.load("model.tflearn")
else:  # Treina a Deep Neural Network
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    # model.save("model.tflearn")  # Salva o modelo


# Função que converte as entradas fornecidas pelo usuário no chat em 0 e 1
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s, language="portuguese")
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():  # Função que implementa o chat
    print("Inicie a conversa com o Bot (digite sair para parar)!")
    while True:
        inp = input("Você: ")
        if inp.lower() == "sair":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]

            print("Bot: " + random.choice(responses))
        else:
            print(
                "Bot: Me desculpe, não sei te dizer! Tente novamente ou pergunte outra coisa. :)")


chat()
