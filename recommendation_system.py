import logging
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from database.database import load_recommendation_history, save_recommendation_history
from ai_tools.analyze_system import extract_tags_and_genres
from ai_tools.summarize_system import summarize_text_and_update_db as compress_text
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('recommendation_debug.log')
    ]
)
logger = logging.getLogger(__name__)

similarity_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def log_call(func):
    """Декоратор для логирования вызовов функций"""

    def wrapper(*args, **kwargs):
        logger.debug(f"Вызов функции {func.__name__} с args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Функция {func.__name__} завершилась успешно")
            return result
        except Exception as e:
            logger.error(f"Ошибка в функции {func.__name__}: {str(e)}", exc_info=True)
            raise

    return wrapper

def preprocess_text(text: str) -> str:
    """Очистка текста для сравнения"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^а-яё\s]', '', text)
    stopwords = {"это", "как", "что", "книга", "роман"}
    return " ".join([word for word in text.split() if word not in stopwords])


@log_call
def find_related_books(query: str, dataset: pd.DataFrame, count: int) -> List[Dict]:
    """Находит связанные книги по расширенным критериям"""

    query_lower = query.lower()
    category_matches = dataset["category"].str.lower().str.contains(query_lower, na=False)
    desc_matches = dataset["description"].str.lower().str.contains(query_lower, na=False)

    combined_matches = dataset[category_matches | desc_matches]


    if len(combined_matches) < count:
        remaining = count - len(combined_matches)
        random_sample = dataset.sample(min(len(dataset), remaining))
        combined_matches = pd.concat([combined_matches, random_sample])

    return combined_matches.head(count).to_dict("records")


def recommend_books_by_genres_or_text(user_text, books_dataset):
    """Добавленная функция рекомендаций по жанрам"""
    user_genres = extract_tags_and_genres(user_text)
    genre_candidates = []

    for book in books_dataset:
        book_genres = book.get("genres", [])

        if book_genres:
            user_genre_embeddings = similarity_model.encode(user_genres, convert_to_tensor=True)
            book_genre_embeddings = similarity_model.encode(book_genres, convert_to_tensor=True)
            cosine_scores = util.cos_sim(user_genre_embeddings, book_genre_embeddings)

            if cosine_scores.max().item() > 0.6:
                genre_candidates.append(book)

    if genre_candidates:
        book_texts = [b["description"] for b in genre_candidates]
        similarities = get_semantic_similarities(user_text, book_texts)
        sorted_books = [book for _, book in
                        sorted(zip(similarities, genre_candidates), key=lambda x: x[0], reverse=True)]
        return [{"name": b["name"], "author": b["author"], "description": b["description"]}
                for b in sorted_books[:5]]
    else:
        all_texts = [b["description"] for b in books_dataset]
        similarities = get_semantic_similarities(user_text, all_texts)
        sorted_books = [book for _, book in sorted(zip(similarities, books_dataset), key=lambda x: x[0], reverse=True)]
        return [{"name": b["name"], "author": b["author"], "description": b["description"]}
                for b in sorted_books[:5]]


def get_semantic_similarities(user_text, texts):
    """Добавленная функция вычисления семантического сходства"""
    user_embedding = similarity_model.encode(user_text, convert_to_tensor=True)
    book_embeddings = similarity_model.encode(texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, book_embeddings)[0]
    return cosine_scores.cpu().tolist()


@log_call
def format_books(book_list):
    """Форматирует список книг в красивый вид с описаниями и разделителями."""
    logger.debug(f"Форматирование {len(book_list)} книг")

    if not book_list:
        return "К сожалению, не удалось найти подходящих рекомендаций 😔"

    formatted_books = ["✨ 📚 *Рекомендуемые книги* 📚 ✨\n"]

    for i, book in enumerate(book_list, 1):
        title = book.get("name", "Без названия")
        authors = book.get("author", "Без автора").replace("By ", "").strip()
        description = book.get("description", "Описание отсутствует")

        short_description = (description[:250] + '...') if len(description) > 250 else description

        formatted_books.append(
            f"🔹 *{title}*\n"
            f"👤 *Автор:* {authors}\n"
            f"📜 *Описание:* {short_description}\n"
        )

        if i < len(book_list):
            formatted_books.append("\n――――――――――――――――――――――\n")

    return "".join(formatted_books)


@log_call
def compute_similarity(target_text: str, books_dataset: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Безопасное вычисление схожести с проверкой размеров"""
    try:
        if books_dataset.empty:
            logger.warning("Пустой датасет на входе")
            return pd.DataFrame(columns=["name", "author", "description", "similarity"])


        target_clean = preprocess_text(target_text)
        books_dataset["clean_text"] = books_dataset["description"].fillna("").apply(preprocess_text)


        if books_dataset["clean_text"].empty or not target_clean:
            logger.error("Нет текстов для сравнения")
            return books_dataset.head(top_n).assign(similarity=0.0)


        target_embedding = similarity_model.encode(target_clean, convert_to_tensor=True)
        book_embeddings = similarity_model.encode(books_dataset["clean_text"].tolist(), convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(target_embedding, book_embeddings)[0].cpu().numpy()


        result_df = books_dataset.copy()
        result_df["similarity"] = similarities

        return result_df.nlargest(top_n, "similarity")

    except Exception as e:
        logger.error(f"Ошибка в compute_similarity: {str(e)}")
        return books_dataset.head(top_n).assign(similarity=0.0)


@log_call
async def get_preferences_from_input(user_input: str) -> Dict[str, List[str]]:
    """Извлекает теги и жанры из пользовательского ввода."""
    logger.debug(f"Извлечение предпочтений из ввода: '{user_input[:50]}...'")
    extracted_data = await extract_tags_and_genres(user_input)
    logger.debug(f"Извлеченные данные: {extracted_data}")
    return extracted_data


@log_call
async def get_preferences_from_history(selected_history: str) -> Dict[str, List[str]]:
    """Сжимает описание книги и извлекает теги/жанры."""
    logger.debug(f"Обработка истории длиной {len(selected_history)} символов")

    try:
        compressed_text = selected_history
        try:
            compressed_text = await compress_text(
                selected_history,
                book_id=0,
                overall_target_chars=1000
            )
        except Exception as e:
            logger.error(f"Ошибка сжатия текста: {str(e)}")

        extracted_data = await extract_tags_and_genres(compressed_text)
        logger.debug(f"Извлеченные данные: {extracted_data}")
        return {
            "tags": list(set(extracted_data.get("tags", []))),
            "genres": list(set(extracted_data.get("genres", []))),
        }, compressed_text

    except Exception as e:
        logger.error(f"Ошибка при обработке истории: {str(e)}")
        try:
            extracted_data = await extract_tags_and_genres(selected_history)
            return {
                "tags": list(set(extracted_data.get("tags", []))),
                "genres": list(set(extracted_data.get("genres", []))),
            }, selected_history
        except Exception as fallback_error:
            logger.error(f"Даже fallback не сработал: {str(fallback_error)}")
            return {"tags": [], "genres": []}, selected_history


@log_call
def combine_preferences(pref1: Dict[str, List[str]], pref2: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Объединяет теги и жанры."""
    combined = {
        "tags": list(set(pref1.get("tags", []) + pref2.get("tags", []))),
        "genres": list(set(pref1.get("genres", []) + pref2.get("genres", []))),
    }
    logger.debug(f"Объединенные предпочтения: {combined}")
    return combined


@log_call
def filter_books_by_tags(dataset, preferences: List[str], max_candidates: int = 40):
    """Фильтрует книги по тегам в категориях или описании."""
    logger.debug(f"Фильтрация по тегам: {preferences}")

    if not preferences:
        return dataset.sample(min(len(dataset), max_candidates))

    preferences = [tag.lower() for tag in preferences if tag.lower() != "художественный"]

    category_matches = dataset["category"].str.lower().isin(preferences)

    if not category_matches.any():
        description_matches = dataset["description"].str.lower().str.contains('|'.join(preferences))
        filtered = dataset[description_matches]
    else:
        filtered = dataset[category_matches]

    if len(filtered) == 0:
        return dataset.sample(min(len(dataset), max_candidates))

    return filtered.sample(min(len(filtered), max_candidates))


@log_call
async def search_books_1_mode(selected_history: str, dataset: pd.DataFrame) -> List[Dict]:
    try:

        pref_history, compressed_text = await get_preferences_from_history(selected_history)
        preferences = pref_history["tags"] + pref_history["genres"]


        filtered_books = filter_books_by_tags(dataset, preferences)


        recommendations = compute_similarity(
            target_text=compressed_text,
            books_dataset=filtered_books,
            top_n=5
        )


        if recommendations.empty:
            logger.warning("Нет рекомендаций. Возвращаю случайные книги")
            return dataset.sample(min(5, len(dataset)))[["name", "author", "description"]].to_dict("records")

        return recommendations[["name", "author", "description"]].to_dict("records")

    except Exception as e:
        logger.error(f"Ошибка в search_books_1_mode: {str(e)}")
        return dataset.sample(min(5, len(dataset)))[["name", "author", "description"]].to_dict("records")


@log_call
async def search_books_2_mode(user_input: str, dataset: pd.DataFrame) -> List[Dict]:
    try:

        pref_input = await get_preferences_from_input(user_input)
        preferences = pref_input["tags"] + pref_input["genres"]


        filtered_books = filter_books_by_tags(dataset, preferences, max_candidates=100)


        recommendations = compute_similarity(
            target_text=user_input,
            books_dataset=filtered_books,
            top_n=5
        )


        if not recommendations.empty and recommendations.iloc[0]["similarity"] < 0.3:
            logger.warning("Низкое качество рекомендаций. Возвращаю книги по тегам")
            return filtered_books[["name", "author", "description"]].head(5).to_dict("records")

        return recommendations[["name", "author", "description"]].to_dict("records")

    except Exception as e:
        logger.error(f"Ошибка в search_books_2_mode: {str(e)}")
        return dataset.sample(min(5, len(dataset)))[["name", "author", "description"]].to_dict("records")@log_call
async def search_books_3_mode(user_input: str, selected_history: str, dataset: pd.DataFrame) -> List[Dict]:
    try:

        pref_input = await get_preferences_from_input(user_input)
        pref_history, compressed_text = await get_preferences_from_history(selected_history)
        combined_pref = combine_preferences(pref_input, pref_history)


        filtered_books = filter_books_by_tags(
            dataset,
            combined_pref["tags"] + combined_pref["genres"],
            max_candidates=150
        )


        combined_text = f"{user_input} {compressed_text}"


        recommendations = compute_similarity(
            target_text=combined_text,
            books_dataset=filtered_books,
            top_n=5
        )


        if len(recommendations) < 3:
            extra_books = filter_books_by_tags(dataset, pref_input["tags"], 2)
            recommendations = pd.concat([recommendations, extra_books])

        return recommendations[["name", "author", "description"]].to_dict("records")

    except Exception as e:
        logger.error(f"Ошибка в search_books_3_mode: {str(e)}")
        return dataset.sample(min(5, len(dataset)))[["name", "author", "description"]].to_dict("records")

@log_call
async def get_book_recommendations(
        user_input: str,
        selected_book: str,
        mode: int,
        user_id: str,
        history_exclude_option: bool,
        dataset
) -> str:
    """Основная функция для получения рекомендаций."""
    logger.info("=" * 50)
    logger.info(f"НАЧАЛО ОБРАБОТКИ ЗАПРОСА (режим {mode})")
    logger.info(f"Параметры вызова:")
    logger.info(f"- user_input: {user_input[:100]}...")
    logger.info(f"- selected_book: {selected_book[:100] if selected_book else 'None'}")
    logger.info(f"- user_id: {user_id}")
    logger.info(f"- history_exclude_option: {history_exclude_option}")
    logger.info(f"- dataset кол-во записей: {len(dataset)}")

    if history_exclude_option:
        logger.info("Проверка истории рекомендаций...")
        previously_recommended = load_recommendation_history(user_id)
        logger.info(f"Ранее рекомендованные книги: {previously_recommended}")
        dataset = dataset[~dataset["name"].isin(previously_recommended)]
        logger.info(f"Осталось книг после фильтрации: {len(dataset)}")

    try:
        if mode == 1:
            recommendations = await search_books_1_mode(selected_book, dataset)
        elif mode == 2:
            recommendations = await search_books_2_mode(user_input, dataset)
        elif mode == 3:
            recommendations = await search_books_3_mode(user_input, selected_book, dataset)
        else:
            raise ValueError(f"Недопустимый режим: {mode}")

        logger.info(f"Найдено рекомендаций: {len(recommendations)}")
        save_recommendation_history(user_id, [book["name"] for book in recommendations])
        return format_books(recommendations)

    except Exception as e:
        logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}", exc_info=True)
        raise