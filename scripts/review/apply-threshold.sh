#!/bin/sh
# Порог серьёзности и рендер вердикта (схема v2). Публикацией НЕ занимается:
# stdout уходит в `gh pr comment` из CI и в терминал из локального прогона.
# Разрез проходит ровно здесь, иначе в скрипт пришлось бы тащить `gh` и права
# на запись.
#
# БЛОКИРУЮТ только находки, у которых одновременно: severity blocker|major,
# `confidence: high`, непустые `scenario` и `observed_result` и хотя бы один
# элемент `evidence`. Правило владельца (2026-08-23): убедительно звучащая
# гипотеза без проверенного кода не имеет права останавливать мерж; для гейта
# precision важнее полноты. Находка, не добравшая до блокировки, всё равно
# рендерится — с явной пометкой, чего ей не хватило: молча понижать её значило
# бы прятать от человека сигнал, который модель сочла major.
#
# Коды выхода: 0 — блокирующих нет; 1 — есть; 2 — вердикт негоден.
# `pipefail` не используется — его нет в POSIX sh.
set -eu

usage() {
    echo "usage: apply-threshold.sh --verdict <file> [--format markdown|text]" >&2
}

verdict=""
format="markdown"

while [ $# -gt 0 ]; do
    case "$1" in
        # Голый флаг без значения (нет $2) — ошибка конфигурации, а не молчаливый
        # сдвиг мимо края позиционных параметров: без сторожа `shift 2` на bash
        # молча съедает лишнее (exit 1), а на dash падает с сообщением самого
        # shell мимо usage() — платформозависимое поведение опаснее прямого exit 2.
        --verdict) [ $# -ge 2 ] || { usage; exit 2; }; verdict="$2"; shift 2 ;;
        --format)  [ $# -ge 2 ] || { usage; exit 2; }; format="$2"; shift 2 ;;
        *) usage; exit 2 ;;
    esac
done

[ -n "$verdict" ] || { usage; exit 2; }
[ -f "$verdict" ] || { echo "вердикт не найден: $verdict" >&2; exit 2; }

case "$format" in
    markdown|text) ;;
    *) echo "неизвестный --format: $format" >&2; exit 2 ;;
esac

# Префлайт jq — по образцу sha256sum/shasum в build-prompt.sh/checksum.sh:
# отсутствие инструмента — отказ конфигурации (код 2) с названной причиной,
# а не смерть шелла кодом 127 вне документированного контракта 0/1/2.
command -v jq >/dev/null 2>&1 || {
    echo "jq не найден в PATH — сверить и отрендерить вердикт нечем." >&2
    exit 2
}

# Невалидный вердикт — отказ (код 2), а не «замечаний нет». Проверяется ЗДЕСЬ,
# а не доверяется схеме codex: порог ниже читает `severity`/`confidence` как
# allow-list, и значение вне enum'а обязано отвергнуться, а не молча оцениться
# как «не блокирует» — иначе негодный вердикт красится зелёным, инвертируя
# инвариант кита. `type == "object" and (...)` — короткое замыкание jq: на
# элементе-не-объекте не индексируем поля и не падаем случайным кодом jq.
#
# ПРИСУТСТВИЕ полей проверяется наравне с типом, теми же требованиями, что у
# схемы (все поля находки обязательны; у каждого элемента evidence обязательны
# file/line/reason). Терпимость к отсутствию здесь стоила бы инварианта с двух
# сторон сразу, и обе нашёл гейт на этом же PR: находка БЕЗ scenario/evidence
# уходила в мягкое «не блокирует» с кодом 0 — негодный вердикт зелёным; а
# evidence из пустых объектов `{}` проходил как заполненный — и БЛОКИРОВАЛ.
# Различие с «пустой строкой» намеренное: пустая строка — модель ответила и
# сказала «нечего», это понижает блокировку; отсутствие ключа — вердикт вне
# схемы, судить по нему нельзя вовсе.
jq -e '(.findings | type == "array")
       and all(.findings[]; type == "object"
               and (.severity | IN("blocker", "major", "minor", "nit"))
               and (.confidence | IN("high", "medium", "low"))
               and ([.title, .file, .scenario, .observed_result, .expected_result]
                    | all(type == "string"))
               and (.line | type == "number" and . == floor and . >= 0)
               and (.evidence | type == "array")
               and all(.evidence[]; type == "object"
                       and (.file | type == "string")
                       and (.line | type == "number" and . == floor and . >= 0)
                       and (.reason | type == "string")))
       and (.note | type == "string")' \
    "$verdict" >/dev/null 2>&1 \
    || { echo "вердикт нечитаем: находка вне схемы v2 — отсутствующее или нестроковое" \
        "текстовое поле, severity/confidence вне enum, line не число, evidence не" \
        "массив объектов file/line/reason, либо note не строка" >&2; exit 2; }

total=$(jq '.findings | length' "$verdict")

# Одно определение «блокирует» на подсчёт и на рендер: две копии однажды
# разойдутся, и пометка в отчёте перестанет совпадать с кодом выхода.
# «Заполнено» меряется после trim: строка из одних пробелов и evidence с
# пробельным reason — та же пустота, только замаскированная; без trim'а
# blocker/major с бланковым обоснованием блокировал бы мерж (гейт на #98).
# Локация тоже входит в «заполнено»: находка с file:"" (и evidence, у которого
# пустые file) не указывает ни на одну читаемую строку дерева — блокировать ею
# мерж значит красный чек, по которому человеку некуда пойти. `line` нарочно
# НЕ проверяется на >0: 0 — легитимный указатель уровня файла, а не пустота.
BLOCKING_DEF='def blank: ((. // "") | gsub("\\s"; "") | length) == 0;
def missing:
    [ (if (.confidence // "") != "high" then "confidence не high" else empty end),
      (if (.file | blank) then "нет file" else empty end),
      (if (.scenario | blank) then "нет scenario" else empty end),
      (if (.observed_result | blank) then "нет observed_result" else empty end),
      (if ([(.evidence // [])[]
            | select((.reason | blank | not) and (.file | blank | not))] | length) == 0
       then "нет evidence" else empty end) ];
def blocking: (.severity | IN("blocker", "major")) and (missing | length == 0);'

blocking=$(jq "$BLOCKING_DEF"'[.findings[] | select(blocking)] | length' "$verdict")

if [ "$format" = markdown ]; then
    echo "## Ревью Codex — независимый чек"
else
    echo "Ревью Codex — независимый чек"
fi
echo
jq -r '.note // ""' "$verdict"
echo

if [ "$total" = 0 ]; then
    echo "Находок нет."
elif [ "$format" = markdown ]; then
    # Блок на находку, не таблица: v2-поля в строку таблицы не помещаются, а
    # обрезать evidence ради ширины значило бы прятать ровно то, ради чего он
    # введён. Текст пишет модель — переводы строк схлопываются, чтобы markdown
    # не разъезжался.
    jq -r "$BLOCKING_DEF"'
        def cell: (. // "") | tostring | gsub("\r?\n"; " ");
        def ev: [.evidence[]? | "`\(.file|cell):\(.line|cell)` — \(.reason|cell)"]
                | join("; ");
        def gate: if (.severity | IN("blocker", "major")) | not then
                      "не блокирует по severity"
                  elif blocking then "БЛОКИРУЕТ"
                  else "не блокирует: " + (missing | join(", ")) end;
        .findings[]
        | "### [\(.severity|cell)] \(.title|cell) — `\(.file|cell):\(.line|cell)`\n"
          + "- Сценарий: \(.scenario|cell)\n"
          + "- Наблюдаемое: \(.observed_result|cell)\n"
          + "- Ожидаемое: \(.expected_result|cell)\n"
          + "- Evidence: \(if (.evidence // []) | length == 0 then "—" else ev end)\n"
          + "- confidence: \(.confidence|cell) → \(gate)\n"
    ' "$verdict"
else
    jq -r "$BLOCKING_DEF"'
        def cell: (. // "") | tostring | gsub("\r?\n"; " ");
        def gate: if (.severity | IN("blocker", "major")) | not then
                      "не блокирует по severity"
                  elif blocking then "БЛОКИРУЕТ"
                  else "не блокирует: " + (missing | join(", ")) end;
        .findings[]
        | "[\(.severity|cell)/\(.confidence|cell)] \(.title|cell) (\(.file|cell):\(.line|cell))\n"
          + "    сценарий: \(.scenario|cell)\n"
          + "    наблюдаемое: \(.observed_result|cell); ожидаемое: \(.expected_result|cell)\n"
          + "    evidence: \([.evidence[]? | "\(.file|cell):\(.line|cell) — \(.reason|cell)"] | join("; "))\n"
          + "    \(gate)\n"
    ' "$verdict"
fi

echo
echo "_Порог: красным делают только \`blocker\`/\`major\` с \`confidence: high\`"
echo "и заполненными scenario, observed_result и evidence. Это чек, не аппрув —"
echo "и не замена ревью человека._"

[ "$blocking" -eq 0 ] || exit 1
