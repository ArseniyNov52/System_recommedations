import requests
import csv
import time

category_translation = {
    "Fiction": "Художественная литература",
    "Science": "Научная литература",
    # ... остальные переводы из предыдущего кода ...
}

def translate_category(tag):
    return category_translation.get(tag.strip(), tag.strip())

def fetch_books(query, start_index, lang="ru", max_results=40):
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&startIndex={start_index}&maxResults={max_results}&langRestrict={lang}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("items", [])
        print(f"Ошибка {response.status_code} для запроса '{query}'")
        return []
    except Exception as e:
        print(f"Ошибка соединения: {e}")
        return []

def process_book(book):
    volume_info = book.get("volumeInfo", {})
    if volume_info.get("language", "").lower() != "ru":
        return None

    title = volume_info.get("title", "").strip()
    authors = ", ".join(a.strip() for a in volume_info.get("authors", [])) if volume_info.get("authors") else ""
    description = volume_info.get("description", "").strip()

    if not title or not authors or not description:
        return None

    # Определяем основную категорию
    api_categories = volume_info.get("categories", [])
    if isinstance(api_categories, str):
        api_categories = [api_categories]

    main_tag = translate_category(api_categories[0]) if api_categories else ""

    # Если категория не определена, используем общую
    if not main_tag:
        main_tag = "Художественная литература" if "fiction" in str(api_categories).lower() else "Научная литература"

    return [title, authors, main_tag, description]

def main():
    # Подзапросы для двух основных категорий
    fiction_subqueries = [
    # Жанры
    "фантастика", "фэнтези", "детектив", "триллер", "ужасы", "роман",
    "любовный роман", "исторический роман", "приключения", "мистика",
    # Формы
    "рассказы", "повести", "поэзия", "драма", "пьесы",
    # Периоды
    "классика", "современная литература", "литература 19 века",
    "литература 20 века", "литература 21 века",
    # Демография
    "детская литература", "подростковая литература", "литература для взрослых",
    # Национальные
    "русская литература", "зарубежная литература",
    # Дополнительные
    "биография", "мемуары", "сказки", "басни", "антиутопия"
]

    science_subqueries = [
    # Основные науки
    "физика", "математика", "химия", "биология", "астрономия",
    "география", "геология", "экология",
    # Гуманитарные
    "история", "философия", "психология", "социология",
    "политология", "экономика", "правоведение",
    # Технические
    "информатика", "программирование", "инженерия",
    "робототехника", "искусственный интеллект",
    # Медицина
    "медицина", "анатомия", "физиология", "генетика",
    # Прикладные
    "бизнес", "менеджмент", "маркетинг", "финансы",
    "педагогика", "языкознание", "культурология"
]

    queries = {
        "Художественная литература": fiction_subqueries,
        "Научная литература": science_subqueries
    }

    unique_ids = set()
    books_data = []
    target = 10000  # Желаемое количество книг
    max_per_query = 3000  # Лимит книг на подзапрос

    print("Начинаем сбор данных...")

    for main_category, subqueries in queries.items():
        for subquery in subqueries:
            total_from_subquery = 0

            for start_index in range(0, max_per_query, 40):
                if len(unique_ids) >= target:
                    break

                items = fetch_books(subquery, start_index)
                if not items:
                    break

                for book in items:
                    book_id = book.get("id")
                    if not book_id or book_id in unique_ids:
                        continue

                    processed = process_book(book)
                    if not processed:
                        continue

                    # Заменяем категорию на основную (художественная/научная)
                    processed[2] = main_category

                    unique_ids.add(book_id)
                    books_data.append(processed)
                    total_from_subquery += 1

                    if len(unique_ids) >= target:
                        break

                print(f"{main_category} ({subquery}): собрано {len(unique_ids)} книг")
                time.sleep(1)  # Чтобы не превысить лимиты API

            print(f"По подзапросу '{subquery}' найдено {total_from_subquery} книг")

    print(f"\nСбор завершён. Всего книг: {len(unique_ids)}")

    with open("books_simple_categories.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Название", "Автор", "Категория", "Описание"])
        writer.writerows(books_data)

    print("Файл успешно сохранён!")
