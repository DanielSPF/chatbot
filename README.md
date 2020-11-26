# Chatbot

Projeto Final da Disciplina de Inteligência Artificial do curso de Sistemas de Informação - UNIMONTES

## Base do Projeto - Tutorial Tech With Tim

This deep learning chatbot tutorial will show you how to use our previously created chatbot model to make predictions and chat back and forth with our user.

- [Text-Based Tutorial](https://www.techwithtim.net/tutorials/ai-chatbot/)

- [Playlist](https://www.youtube.com/watch?v=wypVcNIH6D4&list=PLzMcBGfZo4-ndH9FoC4YWHGXG5RZekt-Q)

- [References](https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077)

# Instalação

Para concluir a instalação é preciso ter instalado em sua máquina o [Anaconda](https://www.anaconda.com/) e como Editor de Texto foi usado o [VS Code](https://code.visualstudio.com/).

Clone o repositório:

Com HTTPS:

```bash
    git clone https://github.com/DanielSPF/chatbot.git
```

Com SSH:

```bash
    git clone git@github.com:DanielSPF/chatbot.git
```

Acesse a pasta do projeto:

```bash
   cd chatbot/
```

Crie o ambiente virtual:

```bash
    conda create -n chatbot python=3.6
```

Ative o ambiente:

```bash
    conda activate chatbot
```

Instale as dependências:

```bash
    pip install nltk numpy tflearn tensorflow==1.14
```

Rode o programa:

```bash
    python main.py
```
