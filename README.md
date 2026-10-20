# Conductivity fluctuation analysis

Обработка бинарных Q00, признаки спектра и групповая оценка классификации Bi/Cr/Fe.

## Структура

- src/dataset_config.py — общий реестр файлов, каналов и настроек.
- src/signal_processing.py — чтение Q00 и построение CSV.
- src/train_models.py — оценка моделей, сводки и графики.
- notebooks/01_signal_processing.ipynb — исследование обработки.
- notebooks/02_train_models.ipynb — оценка моделей.

build_dataset.py не используется и не нужен.

## Что зарегистрировано

- 24 файла Bi/Cr/Fe: по два канала 12 серий.
- Шесть файлов XXX01C01–C06: reference, без автоматического включения в обучение.
- CO105C01/C02, AL103C01/C02 и COB01C01/C02/C03: exploratory.

Отсутствующие зарегистрированные файлы пропускаются и отмечаются в file_inventory.csv.
Ag, Pt и другие серии не добавлены автоматически: нужны согласованные метаданные.

Входные файлы можно расположить непосредственно в data/q00/ или в подпапках годов.
Дубли одного имени в нескольких подпапках вызывают ошибку.

## Разметка и ограничения

baseline — прежние серии 2005 года Bi/Cr/Fe, используемые как аналитическая обучающая база.
comparison — прежние серии 2006 года. Это НЕ доказанный статус облучения.
irradiated=None до подтверждения журналом; выставляйте True/False только по документам.
used_for_training сохраняет прежний аналитический отбор, но не утверждает физическую стабильность.
Если такой отбор не обоснован журналом, оценку нельзя интерпретировать как распознавание стабильных плёнок.

C01/C02 Bi/Cr/Fe обозначены разными каналами; режим регулирования неизвестен.
Для CO105 и AL103 режимы взяты из описания: C01 unregulated, C02 regulated.
year — метка каталога исходной таблицы, не вычисленный календарный год из JD.
Даты из прежних ответов не перенесены: они не были надёжно проверены.

experiment_id объединяет каналы одной серии. Независимость серий/образцов требует проверки.
Если несколько серий относятся к одному физическому образцу, нужен более широкий идентификатор группы.
Синхронные кампании разных материалов могут требовать отдельного разделения по сеансам.

## Признаки

Окно 128 точек, перекрытие 50%, удаление линейного тренда, Blackman и rFFT.
gamma — знаковый наклон log(power) от log(frequency).
energy — среднее положительных ненулевых частотных компонент |FFT|², как в предыдущем коде;
это не нормированная PSD и не интегральная физическая энергия.
log_energy = log10(energy); недопустимые значения не заменяются искусственным минимумом.
Частоты из таблицы являются средней оценкой по файлу; постоянство шага внутри записей нужно проверить.

В CSV добавлены experiment_id, channel, record_index, sample_start,
analysis_role, control_mode, measurement_type, electrode_distance_mm.
Расстояние электродов COB01 размечено только в C03 по номерам записей, начиная с 1.

## Установка и запуск (PowerShell, из корня проекта)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/signal_processing.py
.\.venv\Scripts\python.exe src/train_models.py
.\.venv\Scripts\python.exe -m notebook
```

Команды не требуют активации окружения или изменения ExecutionPolicy.
Ноутбуки открывайте из этого сервера, используя ядро установленного окружения.
После замены скриптов перезапустите ядра ранее открытых ноутбуков.
Первый ноутбук по умолчанию загружает готовый CSV; для пересборки установите REBUILD=True.

## Оценка

StratifiedGroupKFold по experiment_id, сейчас 2 fold для двух baseline-серий каждого класса.
Каналы одной серии не пересекаются между train/test. StandardScaler обучается внутри fold.
Точность считается по окнам, не по независимым физическим образцам.
MLP early_stopping=False: нет скрытого случайного разбиения связанных окон для ранней остановки.
После оценки финальный MLP обучается на всех baseline-строках.
Дополнительные материалы пока не включаются в общую модель.

## Выходные файлы

- data/processed/conductivity_features.csv
- data/processed/file_inventory.csv
- data/processed/*_features.csv
- results/metrics/model_metrics.csv — классификация по fold.
- results/metrics/evaluation_splits.csv — состав групп.
- results/metrics/out_of_fold_predictions.csv
- results/metrics/knn_classification_report.csv
- results/metrics/mlp_classification_report.csv
- results/metrics/clustering_metrics.csv — silhouette отдельно от accuracy.
- results/metrics/comparison_film_predictions.csv — все окна сравниваемых серий.
- results/metrics/comparison_file_summary.csv — одна строка на файл.
- results/metrics/irradiated_film_predictions.csv — только подтверждённо облучённые строки.
- results/metrics/file_feature_summary.csv
- results/figures/knn_confusion_matrix.png
- results/figures/mlp_confusion_matrix.png
- results/figures/kmeans_clusters.png
- results/figures/file_level_gamma_comparison.png
- results/figures/file_level_energy_comparison.png
- results/figures/gamma_boxplots_by_file.png
- results/figures/energy_boxplots_by_file.png

Графики разделены по материалу и каналу. Панели имеют независимые y-шкалы — сравнивайте подписи осей.
Error bars показывают SD окон внутри файла, не доверительные интервалы.
Boxplot скрывает выбросы только на рисунке; исходные данные не удаляются.
Высокая вероятность класса не является доказательством отсутствия радиационного эффекта.
Старые изображения из прежних запусков автоматически не удаляются.

## Проверка перед научной интерпретацией

Нужно подтвердить калибровку и структуру Q00, режимы каналов Bi/Cr/Fe,
независимость образцов, статусы и моменты облучения, постоянство временного шага.
Код проверен синтаксически; обработка ваших бинарных файлов и обучение в этой поставке не запускались.
