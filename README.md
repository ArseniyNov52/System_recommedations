# System_recommendations

## Система рекомендаций книг для чат-бота с ИИ-ассистентом: Обзор и функциональность

### 1. Введение: Система рекомендаций для чат-бота с ИИ-ассистентом

Данный репозиторий представляет собой подробное описание системы рекомендаций книг, разработанной в рамках более крупного проекта — «Чат-бот-ассистент для чтения книг с ИИ-ассистентом». Этот проект направлен на создание интеллектуального помощника, который сделает процесс чтения книг более удобным. Чат-бот включает в себя несколько ключевых компонентов, реализуемых различными представителями команды, и системные рекомендации являются одними из основных элементов этой экосистемы. Ее основная задача состоит в том, чтобы помочь открыть для себя новые книги, учитывая интересы и предпочтения пользователя, тем самым решая проблему выбора среди огромного количества доступной литературы.

Система анализа содержания книг: [analyze_system](https://github.com/rajecc/analyze_system) (Автор: Коротаев Никита)

Система сжатия книг: [bc_code](https://github.com/snthnk/bc_code.git) (Автор: Новожилов Арсений)

---

### 2. Ключевые особенности и режимы работы

#### Режим 1 — Рекомендации на основе истории прочитанных книг:
Этот режим позволяет пользователю получать рекомендации, основываясь на анализе ранее прочитанных книг. Процесс работы включает следующие этапы:

1. Пользователь выбирает ранее прочитанные книги.
2. История обрабатывается: тексты сокращаются до описаний.
3. Извлекаются теги и жанры.
4. Производится фильтрация книг из базы по схожим тегам.
5. Проводится семантический анализ на сходство описаний.
6. Выдаются лучшие совпадения.

#### Режим 2 — Рекомендации по описанию желаемой книги:
Этот режим предназначен для ситуаций, когда пользователь представляет, о чём он хочет прочитать, но не знает такую книгу и не может найти. Процесс работы выглядит следующим образом:

1. Пользователь вводит текст о желаемой книге.
2. Из текста извлекаются жанры и теги.
3. Производится фильтрация книг с совпадающими тегами.
4. Выполняется семантическое сравнение текста с описаниями книг.
5. Система предлагает наиболее подходящие книги.

#### Режим 3 — Персонализированные рекомендации:
В этом режиме предусмотрены возможности двух предыдущих, позволяющие пользователю получить наиболее точные и персонализированные рекомендации, учитывая его прошлый читательский опыт, а также текущие настройки. Процесс работы включает в себя следующие этапы:

1. Пользователь вводит текст и выбирает книгу из истории прочитанных.
2. Из обоих источников извлекаются признаки.
3. Все признаки объединяются (жанры, теги).
4. Идёт фильтрация и сравнение книг из базы по комбинированным признакам.
5. Выдаются топовые рекомендации.

---

### 3. Структура проекта

Проект представляет собой систему рекомендаций книг, основанную на семантическом анализе текста, и состоит из трех основных модулей.

#### 1. `test_models.py`: Оценка моделей семантического представления предложений

Скрипт предназначен для сравнения производительности различных предварительно обученных моделей, генерирующих векторные представления предложений (эмбеддинги). Цель — определить модель, наиболее точно отражающую семантическую близость между текстами, что важно для рекомендательной системы. В процессе работы используются библиотеки:
- `sentence_transformers` для загрузки моделей
- `datasets` для набора данных "sick" (оценка семантической схожести)
- `PyTorch` для тензорных вычислений и оценки качества (STS-оценка)
- `pandas` для организации и сохранения результатов
- `tqdm`, `time` и `psutil` для отслеживания прогресса, времени и использования ресурсов

Скрипт загружает модели, кодирует предложения, вычисляет их сходство и оценивает качество полученных представлений на основе STS-оценки.

#### 2. `recommendation_system.py`: Разработка механизма рекомендаций книг

Модуль реализует основную логику рекомендательной системы. Он предоставляет рекомендации на основе семантического анализа запросов пользователя или истории его взаимодействия с системой. Используются библиотеки:
- `sentence_transformers` для вычисления семантического сходства
- `pandas` для работы с данными о книгах
- `logging` для протоколирования
- Предполагаемые модули для взаимодействия с базой данных и анализа текста (`database.database`, `ai_tools`)

Модуль включает функции для предварительной обработки текста, поиска и фильтрации книг, вычисления семантического сходства и форматирования результатов рекомендаций. Поддерживаются различные режимы рекомендаций, учитывающие контекст пользователя.

#### 3. `parsing.py`: Процесс сбора данных о книгах

Скрипт автоматизирует сбор информации о книгах из Google Books API и сохраняет данные в формате CSV. Полученный набор данных используется рекомендательной системой. В работе используются библиотеки:
- `requests` для взаимодействия с API
- `csv` для записи данных
- `time` для контроля запросов
- `sys` для вывода прогресса
- `datetime` для временных меток
- `re` для обработки текста

Скрипт выполняет поиск книг по заданным категориям и запросам, извлекает необходимую информацию, переводит категории и сохраняет данные в файл.
