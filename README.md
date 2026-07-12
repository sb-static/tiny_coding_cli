# Tiny Coding CLI

Минималистичный автономный coding-agent для работы с локальным проектом через любой API, совместимый с OpenAI Chat Completions.

CLI передаёт модели задачу, рабочую директорию и набор инструментов. Модель может читать и изменять файлы, выполнять команды, запускать тесты, искать информацию в интернете и повторять цикл «анализ → действие → проверка» до получения финального ответа.

> ⚠️ Агент способен изменять и удалять файлы, а также выполнять shell-команды. Запускайте его только в отдельной рабочей директории или репозитории под системой контроля версий.

## Возможности

- Подключение к OpenAI-compatible endpoint `/v1/chat/completions`.
- Поддержка native function calling и текстового JSON fallback для tool calls.
- Автономный цикл работы с ограничением числа итераций.
- Безопасные файловые операции внутри заданного `--workdir`.
- Выполнение команд в терминале с таймаутом и захватом `stdout`/`stderr`.
- Встроенные `web_search` и `web_fetch`.
- Автоматическая загрузка инструкций из корневого `AGENTS.md`.
- Выбор набора доступных инструментов через `--tools`.
- Повтор API-запросов при сетевых ошибках, `429` и временных ошибках сервера.
- JSONL-логирование всех шагов агента.
- Ограничение и аккуратное усечение слишком больших результатов.
- Защита от выхода файловых операций за пределы workspace.
- SSRF-защита для `web_fetch` по умолчанию.

## Как это работает

1. CLI создаёт системный prompt для coding-agent с учётом выбранных инструментов.
2. В prompt передаются абсолютный путь workspace, содержимое `AGENTS.md` и задача пользователя.
3. Модель отвечает текстом или вызывает один либо несколько инструментов.
4. CLI выполняет инструменты локально и возвращает результаты модели.
5. Цикл продолжается, пока модель не даст обычный финальный ответ или не будет достигнут `--max-iterations`.

```text
User task
   │
   ▼
OpenAI-compatible model
   │
   ├── read/edit/write files
   ├── run terminal commands
   ├── search/fetch web pages
   │
   ▼
Tool results → model → next action
   │
   ▼
Final answer
```

## Требования

- Python 3.10 или новее.
- Linux, macOS или WSL рекомендуется для terminal-инструмента.
- Пакет [`requests`](https://pypi.org/project/requests/).
- API с совместимым методом `POST /v1/chat/completions`.
- Для полноценной работы инструментов модель или прокси должны поддерживать OpenAI-style function calling.

## Установка

Клонируйте репозиторий и установите зависимость:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install requests
```

Сохраните CLI как `tiny_coding_cli.py` или измените имя файла в командах ниже.

## Быстрый старт

Запустите локальный OpenAI-compatible endpoint, затем передайте задачу и рабочую директорию:

```bash
python tiny_coding_cli.py \
  --task "Создай hello.py, запусти его и проверь результат" \
  --workdir /tmp/demo-project \
  --base-url http://localhost:8000/v1 \
  --api-key test-key \
  --model moa \
  --tools all \
  --verbose
```

По умолчанию используются:

```text
model:        moa
base URL:     http://localhost:8000/v1
API key:      test-key
tools:        all
iterations:   20
temperature:  0.2
```

## Настройка через переменные окружения

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="test-key"
export CODING_CLI_MODEL="moa"
export CODING_CLI_TOOLS="all"
export CODING_CLI_LOG_FILE="./logs/agent.jsonl"
```

После этого команда становится короче:

```bash
python tiny_coding_cli.py \
  --task "Исправь падающие тесты и объясни изменения" \
  --workdir ./project
```

### Поддерживаемые переменные

| Переменная | Назначение | Значение по умолчанию |
|---|---|---|
| `OPENAI_BASE_URL` | Базовый URL API | `http://localhost:8000/v1` |
| `OPENAI_API_KEY` | Bearer token | `test-key` |
| `CODING_CLI_MODEL` | Имя модели | `moa` |
| `CODING_CLI_TOOLS` | Набор инструментов | `all` |
| `CODING_CLI_LOG_FILE` | Путь к JSONL-логу | не задан |

## Параметры командной строки

| Параметр | Обязательный | По умолчанию | Описание |
|---|:---:|---|---|
| `--task` | да | — | Задача для coding-agent. |
| `--workdir` | да | — | Рабочая директория. Создаётся автоматически, если отсутствует. |
| `--model` | нет | `moa` | Имя модели, передаваемое API. |
| `--tools` | нет | `all` | Предоставляемый модели набор инструментов. |
| `--base-url` | нет | `http://localhost:8000/v1` | Базовый URL OpenAI-compatible API. |
| `--api-key` | нет | `test-key` | API key для заголовка `Authorization: Bearer ...`. |
| `--max-iterations` | нет | `20` | Максимальное число вызовов модели. |
| `--command-timeout` | нет | `60` | Стандартный таймаут terminal-команд, секунд. |
| `--api-timeout` | нет | `300` | Таймаут одного API-запроса, секунд. |
| `--temperature` | нет | `0.2` | Температура генерации. |
| `--max-output-chars` | нет | `24000` | Максимальный размер результата одного инструмента. |
| `--web-timeout` | нет | `15` | Таймаут запросов `web_search` и `web_fetch`. |
| `--web-max-results` | нет | `6` | Стандартное число результатов поиска. |
| `--web-max-chars` | нет | `20000` | Максимальный объём текста страницы от `web_fetch`. |
| `--allow-local-fetch` | нет | выключено | Разрешает `web_fetch` обращаться к private/loopback адресам. |
| `--log-file` | нет | — | Путь к JSONL-транскрипту сессии. |
| `--verbose` | нет | выключено | Показывает вызовы модели и инструментов в `stderr`. |

Полная встроенная справка:

```bash
python tiny_coding_cli.py --help
```

## Режимы инструментов

### Все инструменты

```bash
--tools all
```

Также принимаются псевдонимы `default`, `terminal+files` и `files+terminal`. В текущей реализации они включают полный список инструментов, включая web-инструменты.

### Только терминал

```bash
--tools terminal
```

Псевдоним:

```bash
--tools shell
```

### Только файловые инструменты

```bash
--tools files
```

Поддерживаемые псевдонимы: `file`, `file-tools`, `file_tools`.

### Только web-инструменты

```bash
--tools web
```

Поддерживаемые псевдонимы: `internet`, `search`.

### Без инструментов

```bash
--tools none
```

Модель не сможет просматривать или менять workspace и вернёт только текстовый ответ.

### Пользовательский набор

```bash
--tools read_file,search_files,edit_file,terminal
```

Неизвестное имя инструмента приводит к ошибке до запуска агента.

## Доступные инструменты

### Файловые инструменты

| Инструмент | Назначение |
|---|---|
| `get_workdir` | Возвращает абсолютный путь workspace. |
| `list_files` | Показывает файлы и директории, опционально рекурсивно. |
| `read_file` | Читает UTF-8 файл по диапазону строк или с конца. |
| `search_files` | Ищет подстроку или регулярное выражение по текстовым файлам. |
| `write_file` | Создаёт, перезаписывает или дополняет UTF-8 файл. |
| `edit_file` | Заменяет точный фрагмент текста с защитой от неоднозначной замены. |
| `create_directory` | Создаёт директорию вместе с родительскими директориями. |
| `delete_path` | Удаляет файл или директорию. Для директории требуется `recursive=true`. |

### Терминал

| Инструмент | Назначение |
|---|---|
| `terminal` | Выполняет неинтерактивную shell-команду с `cwd`, равным workspace. |

`stdout` и `stderr` объединяются в результат. При слишком большом выводе сохраняются начало и конец, чтобы не потерять traceback или итог тестов.

### Web

| Инструмент | Назначение |
|---|---|
| `web_search` | Выполняет поиск через HTML-интерфейс DuckDuckGo. |
| `web_fetch` | Загружает HTML, text, JSON или XML и возвращает очищенный текст. |

`web_fetch` не является браузером: JavaScript не исполняется, динамический контент не рендерится.

## Примеры использования

### Создать небольшой проект

```bash
mkdir -p /tmp/tiny-agent-demo

python tiny_coding_cli.py \
  --task "Создай Flask-приложение с GET /health, добавь requirements.txt и проверь синтаксис" \
  --workdir /tmp/tiny-agent-demo \
  --tools all
```

### Исправить тесты в существующем репозитории

```bash
python tiny_coding_cli.py \
  --task "Запусти тесты, найди причину ошибок, исправь код и повторно проверь тесты" \
  --workdir "$PWD" \
  --tools all \
  --max-iterations 30 \
  --command-timeout 180 \
  --verbose
```

### Разрешить только безопасную работу с файлами

```bash
python tiny_coding_cli.py \
  --task "Проанализируй конфигурацию и обнови README. Команды запускать не нужно" \
  --workdir ./project \
  --tools files
```

В этом режиме модель не сможет запускать тесты. Системный prompt предписывает ей указать команды для ручной проверки в финальном ответе.

### Разрешить только чтение и поиск по файлам

```bash
python tiny_coding_cli.py \
  --task "Найди, где реализована авторизация, и опиши поток выполнения" \
  --workdir ./project \
  --tools get_workdir,list_files,read_file,search_files
```

### Использовать только терминал

```bash
python tiny_coding_cli.py \
  --task "Найди причину падения pytest и исправь её" \
  --workdir ./project \
  --tools terminal
```

### Исследование с доступом в интернет

```bash
python tiny_coding_cli.py \
  --task "Проверь актуальную документацию библиотеки, затем обнови пример интеграции" \
  --workdir ./project \
  --tools all \
  --web-max-results 8 \
  --web-max-chars 30000
```

### Записать полный лог работы

```bash
python tiny_coding_cli.py \
  --task "Проведи рефакторинг модуля parser.py и запусти тесты" \
  --workdir ./project \
  --log-file ./logs/refactor-session.jsonl \
  --verbose
```

## Инструкции проекта через `AGENTS.md`

Если в корне `--workdir` находится файл `AGENTS.md`, CLI автоматически читает его и добавляет перед задачей пользователя.

Пример:

```markdown
# AGENTS.md

- Используй Python 3.11.
- Не меняй публичные API без необходимости.
- После изменений запускай `pytest -q`.
- Форматируй Python-код через `ruff format`.
```

Учитывается только первый найденный файл из списка `PROJECT_INSTRUCTION_FILES`. В текущей версии это только корневой `AGENTS.md`.

## JSONL-логирование

При использовании `--log-file` каждая запись сохраняется отдельной JSON-строкой:

```json
{"ts":"2026-07-12T12:00:00","type":"session_start","model":"moa","workspace":"/tmp/project"}
{"ts":"2026-07-12T12:00:01","type":"model_response","iteration":1,"content":"...","tool_calls":[...]}
{"ts":"2026-07-12T12:00:01","type":"tool_call","iteration":1,"name":"read_file","args":{"path":"app.py"}}
{"ts":"2026-07-12T12:00:01","type":"tool_result","iteration":1,"name":"read_file","result":{"ok":true}}
```

Возможные типы записей:

- `session_start`
- `system_prompt`
- `project_instructions`
- `user`
- `model_response`
- `tool_call`
- `tool_result`
- `final`
- `stopped`
- `error`

Лог может содержать исходный prompt, содержимое файлов, команды, ответы API и секреты из вывода инструментов. Не публикуйте его без проверки.

## Совместимость API

CLI отправляет запрос:

```http
POST {base_url}/chat/completions
Content-Type: application/json
Authorization: Bearer {api_key}
```

Основная структура payload:

```json
{
  "model": "moa",
  "messages": [],
  "stream": false,
  "tools": [],
  "tool_choice": "auto",
  "temperature": 0.2
}
```

Endpoint должен возвращать OpenAI-compatible объект с `choices[0].message`.

Предпочтительный формат вызова инструмента:

```json
{
  "tool_calls": [
    {
      "id": "call_123",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": "{\"path\":\"app.py\"}"
      }
    }
  ]
}
```

Также поддерживается ограниченный fallback, когда модель выводит JSON-вызов инструмента как обычный текст.

## Повтор запросов

API-запрос автоматически повторяется до трёх раз при:

- `requests.ConnectionError`;
- `requests.Timeout`;
- HTTP `429`;
- HTTP `500`, `502`, `503`, `504`;
- невалидном JSON в успешном ответе.

Используется экспоненциальная задержка. Для `429` учитывается заголовок `Retry-After`, если он содержит число секунд.

## Модель безопасности

### Изоляция файлов

Все пути нормализуются относительно `--workdir`. Попытка использовать `../`, абсолютный путь вне workspace или символическую ссылку, ведущую наружу, отклоняется как `Path escapes workspace`.

Удаление самого корня workspace запрещено.

### Ограничения поиска по файлам

`search_files`:

- пропускает `.git`, `node_modules`, `__pycache__`, `.venv`, `.mypy_cache`, `.pytest_cache`;
- не читает файлы больше 5 MiB;
- пропускает бинарные и невалидные UTF-8 файлы;
- ограничивает число результатов.

### Ограничение web-доступа

По умолчанию `web_fetch` блокирует:

- `localhost`;
- loopback-адреса;
- private IP ranges;
- link-local, reserved и multicast адреса.

Отключить эту защиту можно явно:

```bash
--allow-local-fetch
```

Используйте этот флаг только в доверенной сети. Проверка адреса выполняется до HTTP-запроса, поэтому для высокорисковых окружений рекомендуется дополнительная сетевая изоляция на уровне контейнера или firewall.

### Terminal-инструмент

Команда выполняется с `shell=True` и правами текущего пользователя. CLI не предоставляет sandbox для shell-команд.

Рекомендации:

- запускайте агент от непривилегированного пользователя;
- используйте контейнер, VM или временную директорию;
- храните проект в Git;
- не передавайте агенту production-секреты;
- ограничивайте инструменты через `--tools`;
- проверяйте изменения перед merge.

## Ограничения

- Поддерживаются только текстовые UTF-8 файлы для структурированных file tools.
- `web_search` зависит от HTML-разметки DuckDuckGo и может перестать работать при её изменении или rate limiting.
- `web_fetch` не исполняет JavaScript и не извлекает содержимое PDF или бинарных файлов.
- Нет потоковой генерации: `stream` всегда равен `false`.
- Нет встроенного Git sandbox, подтверждения опасных команд или rollback.
- Несколько tool calls из одного ответа выполняются последовательно.
- При достижении `--max-iterations` CLI возвращает последний текст модели либо сообщение об остановке.
- Значение `CODING_FINAL_OUTPUT` распознаётся только в фактическом выводе terminal-команды; оно не является обязательным для нормального завершения.

## Диагностика

### `Connection refused`

Проверьте, что endpoint запущен и `--base-url` содержит `/v1`:

```bash
curl http://localhost:8000/v1/models
```

### HTTP `401` или `403`

Проверьте API key:

```bash
export OPENAI_API_KEY="your-key"
```

### Модель отвечает текстом вместо tool calls

Убедитесь, что выбранная модель и прокси поддерживают OpenAI function calling и не удаляют поля `tools`, `tool_choice` и `tool_calls`.

CLI умеет распознавать некоторые JSON-вызовы из текста, но native tool calls надёжнее.

### `Path escapes workspace`

Модель попыталась обратиться к пути вне `--workdir`. Переместите необходимые файлы в workspace или выберите общий безопасный корень.

### Команда завершилась по таймауту

Увеличьте лимит:

```bash
--command-timeout 300
```

Инструмент рассчитан на неинтерактивные команды. Не запускайте dev-серверы, watchers, `vim`, `nano`, `less` и процессы, ожидающие пользовательский ввод.

### Слишком большой вывод

Увеличьте лимит:

```bash
--max-output-chars 50000
```

Для больших логов, CSV и датасетов лучше попросить агента обработать данные небольшим скриптом и вывести только итоговую статистику.

## Коды завершения

| Код | Значение |
|---:|---|
| `0` | Сессия завершена без необработанной ошибки. |
| `1` | Ошибка конфигурации, API, инструмента верхнего уровня или другая необработанная ошибка. |
| `130` | Выполнение прервано через `Ctrl+C`. |

Обратите внимание: достижение `--max-iterations` само по себе не приводит к коду `1`; CLI печатает последний доступный ответ и завершается с кодом `0`.
