from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import time
import psutil
import pandas as pd
import torch
from tqdm import tqdm

try:
    dataset = load_dataset("sick", split="test")
    print(f"Загружено {len(dataset)} примеров")
except Exception as e:
    print(f"Ошибка загрузки датасета: {e}")
    exit()

models = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-m3",
    "sentence-transformers/all-MiniLM-L12-v2"  
]

results = []

for model_name in models:
    print(f"\n{'='*50}\nОбработка {model_name}...")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Используется устройство: {device}")
        
        start_load = time.time()
        model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder="./model_cache"
        )
        print(f"Модель загружена за {time.time()-start_load:.1f} сек")

        sentences_a = [item["sentence_A"] for item in dataset]
        sentences_b = [item["sentence_B"] for item in dataset]
        gold_scores = [item["relatedness_score"] for item in dataset]
        
        start_process = time.time()
        
        embeddings_a = model.encode(sentences_a, 
                                  batch_size=32,
                                  convert_to_tensor=True,
                                  device=device,
                                  show_progress_bar=True)
        
        embeddings_b = model.encode(sentences_b,
                                  batch_size=32,
                                  convert_to_tensor=True,
                                  device=device,
                                  show_progress_bar=True)
        
        similarities = util.cos_sim(embeddings_a, embeddings_b).diagonal()
        
        sts_score = torch.corrcoef(torch.stack([
            similarities,
            torch.tensor(gold_scores, device=similarities.device)
        ]))[0, 1].item()
        
        process_time = time.time() - start_process
        ram_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
        
        results.append({
            "Модель": model_name.split("/")[-1],
            "STS Score": round(sts_score, 4),
            "Время обработки (sec)": round(time.time() - start_load, 1),
            "Использование RAM (MB)": round(ram_usage, 1)
        })
        
        del model, embeddings_a, embeddings_b
        torch.cuda.empty_cache() if device == "cuda" else None
        
    except Exception as e:
        print(f"Ошибка при обработке {model_name}: {str(e)}")
        continue

if results:
    df = pd.DataFrame(results)
    print("\nРезультаты сравнения моделей:")
    print(df.to_markdown(index=False))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"sts_results_{timestamp}.csv", index=False)
    print(f"\nРезультаты сохранены в sts_results_{timestamp}.csv")
else:
    print("Не удалось получить результаты ни для одной модели")
