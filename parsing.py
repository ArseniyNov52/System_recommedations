import requests
import csv
import time
import sys
from datetime import datetime
import re

TARGET_BOOKS = 10000
MAX_PER_QUERY = 1000
MAX_WORKERS = 5
MIN_DESCRIPTION_LEN = 30
API_DELAY = 0.5
MAX_RETRIES = 3
REQUEST_TIMEOUT = 20
SAVE_INTERVAL = 500

CATEGORY_TRANSLATION = {

    "Fiction": "Художественная литература",
    "Novel": "Роман",
    "Fantasy": "Фэнтези",
    "Science Fiction": "Научная фантастика",
    "Detective": "Детектив",
    "Thriller": "Триллер",
    "Horror": "Ужасы",
    "Romance": "Любовный роман",
    "Historical Fiction": "Исторический роман",
    "Adventure": "Приключения",
    "Mystery": "Мистика",
    "Classic": "Классика",
    "Poetry": "Поэзия",
    "Drama": "Драма",
    "Biography": "Биография",
    "Memoir": "Мемуары",
    "Fairy Tale": "Сказка",
    "Dystopian": "Антиутопия",
    "Young Adult": "Подростковая литература",

    "Science": "Научная литература",
    "Physics": "Физика",
    "Mathematics": "Математика",
    "Chemistry": "Химия",
    "Biology": "Биология",
    "Astronomy": "Астрономия",
    "Geography": "География",
    "Geology": "Геология",
    "Ecology": "Экология",
    "History": "История",
    "Philosophy": "Философия",
    "Psychology": "Психология",
    "Sociology": "Социология",
    "Politics": "Политология",
    "Economics": "Экономика",
    "Law": "Право",
    "Computer Science": "Информатика",
    "Programming": "Программирование",
    "Engineering": "Инженерия",
    "Medicine": "Медицина",
    "Business": "Бизнес",
    "Self-help": "Саморазвитие",
}

SEARCH_QUERIES = {
    "Художественная литература": [
        "фантастика", "фэнтези", "детектив", "триллер", "ужасы", "роман",
        "любовный роман", "исторический роман", "приключения", "мистика",
        "боевик", "нуар", "киберпанк", "постапокалипсис", "стимпанк",
        "рассказы", "повести", "поэзия", "драма", "пьесы", "эссе",
        "классика", "современная литература", "литература 19 века",
        "литература 20 века", "литература 21 века", "средневековая литература",
        "детская литература", "подростковая литература", "литература для взрослых",
        "русская литература", "зарубежная литература", "советская литература",
        "биография", "мемуары", "сказки", "басни", "антиутопия", "утопия",
        "путешествия", "автобиография", "дневники", "письма",
        "Пушкин", "Толстой", "Достоевский", "Чехов", "Булгаков",
        "Стругацкие", "Лем", "Пелевин", "Акунин", "Улицкая"
    ],
    "Научная литература": [
        "физика", "математика", "химия", "биология", "астрономия",
        "география", "геология", "экология", "метеорология", "океанология",
        "история", "философия", "психология", "социология",
        "политология", "экономика", "правоведение", "культурология",
        "религиоведение", "антропология", "археология",
        "информатика", "программирование", "инженерия",
        "робототехника", "искусственный интеллект", "нейросети",
        "блокчейн", "кибербезопасность", "базы данных",
        "медицина", "анатомия", "физиология", "генетика", "вирусология",
        "биохимия", "фармакология", "диетология", "психиатрия",
        "бизнес", "менеджмент", "маркетинг", "финансы", "бухгалтерия",
        "педагогика", "языкознание", "лингвистика", "переводоведение",
        "кулинария", "рукоделие", "ремонт", "строительство", "садоводство",
        "дизайн", "фотография", "рисование", "музыка", "спорт"
    ]
}

def translate_category(tag):
    if not tag:
        return "Разное"

    if tag in CATEGORY_TRANSLATION:
        return CATEGORY_TRANSLATION[tag]

    if "/" in tag:
        translated_parts = []
        for part in tag.split("/"):
            translated_part = CATEGORY_TRANSLATION.get(part.strip(), part.strip())
            translated_parts.append(translated_part)
        return "/".join(translated_parts)

    return tag.strip()

def fetch_books(query, start_index):
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&startIndex={start_index}&maxResults=40&langRestrict=ru"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                if 'items' in data:
                    return data['items']
                return []
            elif response.status_code == 429:
                wait_time = 10 * (attempt + 1)
                print(f"\nПревышен лимит запросов для '{query}'. Ждем {wait_time} сек...")
                time.sleep(wait_time)
                continue
            else:
                print(f"\nОшибка {response.status_code} для запроса '{query}'")
                return []
        except Exception as e:
            print(f"\nОшибка соединения (попытка {attempt + 1}): {e}")
            time.sleep(5 * (attempt + 1))

    print(f"\nНе удалось выполнить запрос после {MAX_RETRIES} попыток: {query}")
    return []

def process_book(book, main_category):
    if not book or not isinstance(book, dict):
        return None

    volume_info = book.get("volumeInfo", {})
    if not volume_info:
        return None

    language = volume_info.get("language", "").lower()
    if language not in ("ru", "rus"):
        return None

    title = str(volume_info.get("title", "")).strip()
    authors = volume_info.get("authors", [])
    if authors and isinstance(authors, list):
        authors = ", ".join(str(a).strip() for a in authors)
    else:
        authors = ""

    description = str(volume_info.get("description", "")).strip()

    if not title or not authors or len(description) < MIN_DESCRIPTION_LEN:
        return None

    api_categories = volume_info.get("categories", [])
    if isinstance(api_categories, str):
        api_categories = [api_categories]

    translated_categories = []
    for cat in api_categories:
        if isinstance(cat, str):
            translated_cat = translate_category(cat)
            if translated_cat:
                translated_categories.append(translated_cat)

    main_tag = translated_categories[0] if translated_categories else main_category

    return {
        "title": title,
        "author": authors,
        "category": main_tag,
        "description": description,
        "all_categories": ", ".join(translated_categories)
    }

def save_books_to_csv(books_data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["title", "author", "category", "description", "all_categories"])
        writer.writeheader()
        writer.writerows(books_data)

def print_progress(completed, total, collected, target, start_time, unique):
    elapsed = datetime.now() - start_time
    progress = min(100, completed / total * 100)
    books_progress = min(100, collected / target * 100)

    sys.stdout.write(
        f"\rЗапросов: {completed}/{total} ({progress:.1f}%) | "
        f"Книг: {collected}/{target} ({books_progress:.1f}%) | "
        f"Уникальных: {unique} | "
        f"Время: {elapsed}"
    )
    sys.stdout.flush()

def main():
    unique_books = set()
    books_data = []
    total_collected = 0
    completed_queries = 0
    should_stop = False

    total_queries = sum(
        len(subqueries) * (MAX_PER_QUERY // 40)
        for subqueries in SEARCH_QUERIES.values()
    )

    start_time = datetime.now()
    last_save = 0

    print("Начинаем сбор данных...\n")
    print(f"Цель: собрать {TARGET_BOOKS} книг")
    print(f"Всего запросов к API: ~{total_queries}")
    print("="*50)

    print_progress(0, total_queries, 0, TARGET_BOOKS, start_time, 0)

    try:
        for main_category, subqueries in SEARCH_QUERIES.items():
            for subquery in subqueries:
                for start_index in range(0, MAX_PER_QUERY, 40):
                    if should_stop:
                        break

                    items = fetch_books(subquery, start_index)
                    completed_queries += 1

                    for book in items:
                        processed = process_book(book, main_category)
                        if not processed:
                            continue

                        norm_title = re.sub(r'\W+', '', processed['title'].lower())
                        norm_author = re.sub(r'\W+', '', processed['author'].lower())
                        book_id = hashlib.md5(f"{norm_title}_{norm_author}".encode()).hexdigest()

                        if book_id not in unique_books:
                            unique_books.add(book_id)
                            books_data.append(processed)
                            total_collected += 1

                            if total_collected - last_save >= SAVE_INTERVAL:
                                timestamp = datetime.now().strftime("b")
                                save_books_to_csv(books_data, f"books_progress_{timestamp}.csv")
                                last_save = total_collected

                            if total_collected >= TARGET_BOOKS:
                                should_stop = True
                                break

                    print_progress(
                        completed_queries, total_queries,
                        total_collected, TARGET_BOOKS,
                        start_time, len(unique_books)
                    )
                    time.sleep(API_DELAY)

                if should_stop:
                    break
            if should_stop:
                break

        duration = datetime.now() - start_time
        print("\n" + "="*50)
        print(f" Сбор завершен! Получено {total_collected} книг за {duration}")
        print(f"Уникальных книг: {len(unique_books)}")

    except KeyboardInterrupt:
        duration = datetime.now() - start_time
        print(f"\n\n Сбор прерван пользователем после {duration}")
        print(f"Успешно собрано {total_collected} книг")

    except Exception as e:
        print(f"\n\n Критическая ошибка: {e}")
        print(f"Удалось собрать {total_collected} книг")

    finally:
        if books_data:
            timestamp = datetime.now().strftime("a")
            filename = f"books_final_{timestamp}.csv"
            save_books_to_csv(books_data, filename)
            print(f" Данные сохранены в файл: {filename}")

if __name__ == "__main__":
    main()
