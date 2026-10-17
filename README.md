# RAG — корпоративная база знаний
## 1. О задаче.
В этой работе строится ванильный RAG по корпоративной базе знаний: 10 статей, 50 вопросов и эталонные ответы разметчиков (golden test set). Цель — собрать фундамент на LangChain (индекс, ретривер, генератор) и измерить, насколько поиск по базе улучшает ответы относительно той же LLM без контекста.

Последовательно строятся:
- zero-shot LLM по API Groq, без ретривера;
- zero-shot локальная LLM через vLLM, без ретривера;
- RAG на LangChain с той же моделью-генератором, что и в одном из zero-shot прогонов;
- оценка через LLM-as-judge.

Качество измеряется метриками ROUGE-1/2/L, BLEU и LLM-as-judge.
## 2. Данные и выводы по EDA
**Источник:** три файла задания в `data/raw/`.

- `articles.json` — 10 статей базы знаний.
- `questions.json` — 50 вопросов к этой базе.
- `ground_truth.json` — эталонные ответы разметчиков.

**Вывод EDA**
1. 10 статей, 50 вопросов и 50 эталонных ответов. Идентификаторы вопросов и ответов совпадают один к одному, пропусков нет.
2. Средняя длина статьи — около 785 символов, вопроса — 63, эталонного ответа — 57. Ответы короткие, поэтому zero-shot просит одно предложение.
3. Тексты на русском: командировки, инцидент ИБ, SLA, продажи, онбординг, резервные копии, инфраструктура, удалённая работа и другие внутренние политики.

## 3. Выводы экспериментов
### Обоснование выбора моделей
Локальная LLM — `Qwen/Qwen2.5-1.5B-Instruct`. Архитектура `Qwen2ForCausalLM` есть в vLLM 0.8.5, веса BF16 занимают около 3.1 ГБ и помещаются в 8 ГБ RTX 4060. В линейке Qwen2.5 нет ровно 1B: 0.5B слишком слабая для честного сравнения с RAG, 3B в BF16 — уже около 6.2 ГБ одних весов и на этой карте впритык. Qwen2.5 обучена на 29 языках, включая русский, а статьи и эталоны русские. Это instruct-модель без режима рассуждения: для ROUGE/BLEU нужен короткий ответ, а не цепочка мыслей.

LLM по API — `qwen/qwen3.8-27b` через Groq. Бесплатные лимиты у чат-моделей одинаковые по цифрам (30 запросов в минуту, 1000 в день, 8000 токенов в минуту, 200 000 токенов в день), но счётчик у каждой модели свой, поэтому генерация не съедает квоту судьи. На общих бенчмарках она сильнее `gpt-oss-20b` при тех же лимитах: GPQA Diamond 89.2% против 71.5%, Humanity's Last Exam без инструментов 30.8% против 10.9%. В конфиге `reasoning_effort: none`, чтобы ответ оставался коротким и не тратил дневной бюджет токенов на скрытые рассуждения.

LLM as judge — `openai/gpt-oss-120b` через Groq: strict JSON Schema и отдельный дневной счётчик. Локальная модель на 8 ГБ роль судьи не тянет.

vLLM вместо transformers — выше throughput за счёт PagedAttention и continuous batching. На Windows vLLM не используется: локальный инференс идёт из WSL.

RAG — ванильная цепочка, не Deep Agent: индекс десяти статей в локальном Qdrant, `VectorStoreRetriever` с поиском по сходству (`top_k: 2`) и тот же `Qwen2.5-1.5B-Instruct` через vLLM. `RecursiveJsonSplitter` не используется: статья уже один короткий текст, резать JSON нечего. Эмбеддер `intfloat/multilingual-e5-small` на CPU, с префиксами `passage:` и `query:`.

### Автоматические метрики (50 вопросов)

Прогоны: zero-shot API `artifacts/zero_shot_api/20261016-140429`, zero-shot local `artifacts/zero_shot_local/20261016-135558`, RAG `artifacts/rag_local/20261017-134211`.

| Модель | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---|---|---|---|
| zero-shot API (`qwen/qwen3.8-27b`) | 0.0000 | 0.0000 | 0.0000 | 0.0064 |
| zero-shot local (`Qwen2.5-1.5B-Instruct`, vLLM) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **RAG local (`Qwen2.5-1.5B-Instruct` + Qdrant)** | **0.7514** | **0.3422** | **0.7308** | **0.3206** |

### LLM-as-judge (50 вопросов, openai/gpt-oss-120b через Groq)

Шкала 1–5. Судья видит вопрос, эталон и ответ модели, статьи не видит. Все 50 примеров каждого прогона оценены, сбоев нет (`artifacts/judge_results.json`).

| Модель | Correctness | Completeness | Conciseness |
|---|---|---|---|
| zero-shot API | 1.00 | 1.00 | 3.30 |
| zero-shot local | 1.00 | 1.00 | 3.44 |
| **RAG local** | **4.56** | **4.48** | **4.80** |

### Выводы
- База закрытая. В промпте zero-shot прямо сказано отвечать «не знаю», если факта нет в вопросе. Обе модели так и сделали на 49 из 50 вопросов, поэтому ROUGE нулевой. Единственный ответ не «не знаю» — q28: и API, и локальная модель назвали HR, эталон — IT Ops. Это галлюцинация, не знание базы.
- Conciseness у zero-shot около 3.3 — не качество ответа, а короткое «не знаю». Correctness 1.00 как раз фиксирует отказ.
- RAG на той же локальной модели поднимает ROUGE-1 с 0 до 0.75 и correctness с 1.00 до 4.56. Оба сигнала совпадают: выигрыш даёт поиск, не размер модели. API-модель в 27B без контекста не лучше 1.5B.
- Судья поставил RAG correctness ниже 5 на 7 вопросах. Три из них — чужая цифра из той же статьи: лимит по Европе подменён лимитом по России (q3, 9000 рублей вместо 160 евро), 6 пользователей — на 94 (q10), окно P3/P4 — на круглосуточную поддержку (q17). Статья в контексте есть, 1.5B выбирает соседний факт. q25 пересказывает цель сделки и теряет «45 дней». q8 называет 09:12 и пропускает дату. q15 теряет «до» у «8 часов». q24 — единственное «не знаю» у RAG: конверсия 31% в ответ не попала.
- Более ранний RAG-прогон `20261017-132646` с формулировкой «кратко, одним предложением» дал ROUGE-1 0.7235 и BLEU 0.3305. Текущий промпт без слова «кратко» ближе к эталону по ROUGE.

**Как ещё можно улучшить пайплайн**
Ошибки сидят внутри статьи, а не между статьями: `top_k: 2` уже приносит нужный документ, модель путает соседние числа. Следующий шаг — резать статью на абзацы или предложения и ранжировать чанки, а не класть в промпт две политики целиком. Поверх этого — reranker и гибридный поиск (плотные векторы плюс ключевые слова вроде «Европа», «P3»). Query rewriting поможет вопросам с отсылкой «в документе», но не разведёт 9000 рублей и 160 евро внутри одного регламента. Для генерации имеет смысл просить дословно перенести число и единицу измерения из контекста: q8 и q15 ломаются на пропущенном уточнении, а не на ретривере.

# 4. Инфраструктура и установка

## 4.1. WSL

Открыть PowerShell от имени администратора, выполнить `wsl --install`, перезагрузить компьютер. Открыть Ubuntu через меню Пуск, при первом запуске задать логин и пароль Linux (не связан с Windows-аккаунтом). Проверить, что WSL видит GPU: `nvidia-smi` — должна показаться RTX 4060 Laptop. Если пусто — обновить драйвер NVIDIA для Windows и перезапустить WSL: `wsl --shutdown` из PowerShell, потом заново открыть Ubuntu.

На этой машине Ubuntu уже установлена, `nvidia-smi` показывает RTX 4060 Laptop. Дистрибутив по умолчанию у `wsl` без флага — `docker-desktop`, поэтому терминал проекта открывать как Ubuntu из меню Пуск или командой `wsl -d Ubuntu`.

## 4.2. Установка uv в WSL

Установить uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`. Добавить в PATH: `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc`. Установить Python 3.12: `uv python install 3.12`. Проверить: `uv --version`.

В Ubuntu uv уже стоит (`~/.local/bin/uv`).

## 4.3. Системные библиотеки NVIDIA

Подключить репозиторий NVIDIA и установить cuDNN и NCCL:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12
sudo apt install -y libnccl2 libnccl-dev
sudo ldconfig
```

Также поставить системный компилятор (нужен Triton для JIT-компиляции CUDA-ядер):

```bash
sudo apt install -y build-essential
```

cuDNN в этой Ubuntu уже установлен (пакеты ставились для `text-gen`).

## 4.4. Настройка окружения под Linux

Задать отдельный каталог для venv, чтобы не конфликтовал с Windows: `echo 'export UV_PROJECT_ENVIRONMENT=".venv_linux"' >> ~/.bashrc && source ~/.bashrc`. Перейти в проект: `cd /mnt/c/Users/Ivan/Documents/projects/RAG` (здесь `/mnt/c/...` — доступ к диску `C:\` из Linux). Синхронизировать зависимости: `uv sync` — uv создаст `.venv_linux` и установит всё из `pyproject.toml`. Установить PyTorch с CUDA: `uv pip install torch --index-url https://download.pytorch.org/whl/cu124`. Проверить GPU: `uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"` — ожидается `2.6.0+cu124`, `True`, `NVIDIA GeForce RTX 4060 Laptop GPU`. Установить vLLM: `uv pip install vllm`. Проверить импорт: `uv run python -c "import vllm; print(vllm.__version__)"`.

Windows-окружение `.venv` для vLLM не используется. Локальный инференс запускать из Ubuntu.

## 4.5. Переменные окружения

Создать `.env` в корне проекта:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGSMITH_PROJECT=rag
```

- `GROQ_API_KEY` — ключ Groq для zero-shot по API и для LLM-as-judge (получить: https://console.groq.com → API Keys → Create API Key). Инструкция к заданию: https://lokismentor.yonote.ru/doc/zadacha-text-generation-GeqTbU4BTG
- `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` — трейс RAG: вопрос, найденные статьи, промпт и ответ (https://smith.langchain.com). Без ключа прогон не падает и трейсы не отправляет.

## 4.6. Загрузка данных

Скрипта скачивания нет: три файла задания копируются в `data/raw/` вручную.

```text
data/raw/articles.json
data/raw/questions.json
data/raw/ground_truth.json
```

# 5. Запуск экспериментов

Оба zero-shot запускать из Ubuntu (`wsl -d Ubuntu`) в каталоге проекта. Локальной модели нужен vLLM и `.venv_linux`. API-прогону нужен `GROQ_API_KEY` в `.env`. Статьи в промпт не попадают: модель видит только вопрос.

## 5.1. Zero-shot по API Groq

`qwen/qwen3.8-27b`, конфиг `configs/experiments/zero_shot_api.yaml`:

```bash
uv run python experiments/run_zero_shot_api.py
```

## 5.2. Zero-shot локальная LLM

`Qwen/Qwen2.5-1.5B-Instruct` через vLLM, конфиг `configs/experiments/zero_shot_local.yaml`. Первый запуск скачает около 3.1 ГБ весов.

```bash
uv run python experiments/run_zero_shot_local.py
```

## 5.3. RAG

Ванильный RAG на локальной модели. Конфиг `configs/experiments/rag_local.yaml`. Скрипт заново собирает коллекцию `articles` в `data/qdrant` и отвечает на 50 вопросов. Генератор — тот же vLLM, что в п. 5.2. Движок V0 включается в коде, отдельно `VLLM_USE_V1=0` писать не нужно. Запускать из Ubuntu.

```bash
uv run python experiments/run_rag_local.py
```

## 5.4. LLM-as-judge

`openai/gpt-oss-120b` сравнивает ответ с вопросом и эталоном. Критерии в `configs/experiments/judge.yaml`: correctness, completeness, conciseness, шкала 1–5. Судья не видит статьи, только golden answer. В `prediction_files` указаны последние прогоны zero-shot (API и локальный) и RAG. Между запросами пауза 4 секунды, чтобы уложиться в 30 RPM. Дневной лимит останавливает прогон и сохраняет уже посчитанное.

```bash
uv run python experiments/run_judge.py
uv run python experiments/run_judge.py --limit 5
```

## 5.5. Артефакты

Каждый эксперимент сохраняет результаты в `artifacts/<exp_name>/<timestamp>/`:

- `config.json` — конфиг запуска.
- `data.jsonl` — вопрос и эталонный ответ, на которых считался прогон.
- `predictions.jsonl` — вопрос, эталон, ответ модели.
- `metrics.json` — ROUGE и BLEU.

Для judge дополнительно:

- `artifacts/judge/<model>.jsonl` — сырые оценки судьи по каждому примеру.
- `artifacts/judge_results.json` — средние оценки по моделям.

# 6. Структура проекта

```
RAG/
├── configs/
│   ├── general_config.yaml              # пути к данным и артефактам
│   └── experiments/
│       ├── zero_shot_local.yaml         # Qwen2.5-1.5B-Instruct
│       ├── zero_shot_api.yaml           # qwen/qwen3.8-27b
│       ├── rag_local.yaml               # Qdrant + та же локальная модель
│       └── judge.yaml                   # gpt-oss-120b, пути к predictions
├── data/
│   ├── raw/                             # articles, questions, ground_truth
│   └── qdrant/                          # локальный индекс, не коммитится
├── experiments/
│   ├── run_zero_shot_local.py
│   ├── run_zero_shot_api.py
│   ├── run_rag_local.py
│   └── run_judge.py
├── src/
│   ├── data/                            # статьи и пары вопрос-ответ
│   ├── retrieval/                       # эмбеддер e5 и индекс Qdrant
│   ├── experiments/                     # runner и RAG-цепочка
│   ├── preprocessing/                   # сборка промпта
│   ├── models/                          # vLLM и Groq генераторы
│   ├── metrics/                         # compute_metrics, llm_judge
│   └── utils/                           # config, logging, artifacts
├── artifacts/                           # результаты экспериментов
├── pyproject.toml
└── README.md
```
