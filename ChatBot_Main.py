import nltk
import random
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split

# BOT_CONFIG = {
#     'intents': {
#         'hello': {
#             'examples': ['Привет!', 'Здравсвуйте!))', 'Хай!!'],
#             'responses': ['Прив!', 'Хеллоу', 'Как жизнь?']
#         },
#         'bye': {
#             'examples': ['Пока!', 'До свиданья!', 'Увидимся!!'],
#             'responses': ['Чао!', 'Будь здоров', 'Сайонара']
#         }
#     },
#     'default_answers': ['Извините, я тупой', 'Переформулируйте, меня еще не обучили']
# } # "знания" бота

with open('CLEAR_BIG_BOT_CONFIG.json', 'r') as f:
    BOT_CONFIG = json.load(f)  # json-файл конфига в переменную BOT_CONFIG


def cleaner(text):  # функция очистки текста
    cleaned_text = ''
    for ch in text.lower():
        if ch in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz ':
            cleaned_text = cleaned_text + ch
    return cleaned_text


def match(text, example):  # гибкая функция сравнения текстов
    return nltk.edit_distance(text, example) / len(example) < 0.4 if len(example) > 0 else False


def get_intent(text):  # функция определения интента текста
    for intent in BOT_CONFIG['intents']:
        if 'examples' in BOT_CONFIG['intents'][intent]:
            for example in BOT_CONFIG['intents'][intent]['examples']:
                if match(cleaner(text), cleaner(example)):
                    return intent


X = []
y = []

for intent in BOT_CONFIG['intents']:
    if 'examples' in BOT_CONFIG['intents'][intent]:
        X += BOT_CONFIG['intents'][intent]['examples']
        y += [intent for i in range(len(BOT_CONFIG['intents'][intent]['examples']))]

# Обучающая выборку для ML-модели

vectorizer = CountVectorizer(preprocessor=cleaner, ngram_range=(1, 3), stop_words=['а', 'и'])
# Создается векторайзер – объект для превращения текста в вектора

vectorizer.fit(X)
X_vect = vectorizer.transform(X)
# Обучается векторайзер на выборке

X_train_vect, X_test_vect, y_train, y_test = train_test_split(X_vect, y, test_size=0.3)
# Разбивается выборка на train и на test

# log_reg = LogisticRegression()
# log_reg.fit(X_train_vect, y_train)
# log_reg.score(X_test_vect, y_test)

# log_reg.score(X_vect, y)

sgd = SGDClassifier()  # Создается модель
# sgd.fit(X_train_vect, y_train) # Обучается модель
# sgd.score(X_test_vect, y_test) # Проверяется качество модели на тестовой выборке
sgd.fit(X_vect, y)

sgd.score(X_vect, y)  # Проверка качества классификации


def get_intent_by_model(text):  # Функция определяющая интент текста с помощью ML-модели
    return sgd.predict(vectorizer.transform([text]))[0]


def bot(text):  # функция бота
    intent = get_intent(text)  # 1. пытается понять намерение сравнением по Левинштейну
    if intent is None:
        intent = get_intent_by_model(text)  # 2. пытается понять намерение с помощью ML-модели
    return random.choice(BOT_CONFIG['intents'][intent]['responses'])

# question = '' # блок для общения с ботом через консоль
# while question not in ['выход', 'выключайся']:
#     question = input()
#     answer = bot(question)
#     print(answer)

# !pip install python-telegram-bot --upgrade библиотека для подключения к телеграму

# Блок прикрепления бота к телеграму


import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Логин
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


def start(update: Update, _: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi!')


def help_command(update: Update, _: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def echo(update: Update, _: CallbackContext) -> None:
    """Echo the user message."""
    text = update.message.text
    print(text)
    reply = bot(text)
    update.message.reply_text(reply)


def main() -> None:  # функция бота для телеграма
    """Start the bot."""

    updater = Updater("1709264459:AAEQpsEPWjQ9cPd1QpXC-yXy4obQE01WEc8")

    # получение диспетчера для обработки
    dispatcher = updater.dispatcher

    # ответы на команды в телеграме
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    # ответ на не команду
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # запуск бота
    updater.start_polling()

    # Работает до нажатия Ctrl-C или получения SIGINT,
    # SIGTERM или SIGABRT.
    updater.idle()


main()
