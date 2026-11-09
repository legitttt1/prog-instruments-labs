import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

ALPHA = 0.08
YEARS = ['2005', '2010', '2015', '2019', '2020', '2021']
PAIR_YEARS = ('2005', '2021')
KRUSKAL_YEAR = '2015'

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ---------------------------------------------------------
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ---------------------------------------------------------
# Читаем исходный файл
df_raw = pd.read_excel('CHISLO_DOCTORS.xlsx', sheet_name='ROSSTAT')

# Извлекаем все строки с регионами и РФ
df_all = df_raw.iloc[3:95, [0, 7, 8, 9, 10, 11, 12]].copy()
df_all.columns = ['Region'] + YEARS
df_all = df_all.dropna(subset=['Region'])
df_all['Region'] = df_all['Region'].astype(str).str.strip().str.replace('\n', ' ')

for y in YEARS:
    df_all[y] = pd.to_numeric(df_all[y], errors='coerce')

# Извлечение данных по РФ в целом (для п. 6)
rf_row = df_all[df_all['Region'].str.contains('Российская Федерация', case=False, na=False)]
if rf_row.empty:
    rf_values = df_all.iloc[0][YEARS]
else:
    rf_values = rf_row.iloc[0][YEARS]

# Списки субъектов для округов Варианта 10 (ПФО, УФО, СФО)
pfo_regions = [
    'Республика Башкортостан', 'Республика Марий Эл', 'Республика Мордовия',
    'Республика Татарстан', 'Удмуртская Республика', 'Чувашская Республика',
    'Пермский край', 'Кировская область', 'Нижегородская область',
    'Оренбургская область', 'Пензенская область', 'Самарская область',
    'Саратовская область', 'Ульяновская область'
]

ufo_regions = [
    'Курганская область', 'Свердловская область',
    'Тюменская область без автономных округов', 'Челябинская область',
    'Ханты-Мансийский автономный округ – Югра', 'Ямало-Ненецкий автономный округ'
]

sfo_regions = [
    'Республика Алтай', 'Республика Тыва', 'Республика Хакасия',
    'Алтайский край', 'Красноярский край', 'Иркутская область',
    'Кемеровская область', 'Новосибирская область', 'Омская область',
    'Томская область'
]


def filter_by_regions(df, reg_list):
    norm_target = [r.replace(' ', '').replace('–', '-').lower() for r in reg_list]
    df_temp = df.copy()
    df_temp['norm'] = df_temp['Region'].str.replace(' ', '').str.replace('–', '-').str.lower()
    return df_temp[df_temp['norm'].isin(norm_target)].drop(columns=['norm']).reset_index(drop=True)


df_pfo = filter_by_regions(df_all, pfo_regions)
df_ufo = filter_by_regions(df_all, ufo_regions)
df_sfo = filter_by_regions(df_all, sfo_regions)

# ---------------------------------------------------------
# ЧАСТЬ I. ПУНКТ 2: ОПИСАТЕЛЬНАЯ СТАТИСТИКА И ВИЗУАЛИЗАЦИЯ (ПФО)
# ---------------------------------------------------------
print("=" * 70)
print("ПУНКТ 2: ОПИСАТЕЛЬНАЯ СТАТИСТИКА ДЛЯ ПФО")
print("=" * 70)

desc_stats = []
for y in YEARS:
    data = df_pfo[y].dropna()
    desc_stats.append({
        'Год': y,
        'Среднее': data.mean(),
        'Стд. откл.': data.std(ddof=1),
        'Медиана': data.median(),
        'Асимметрия': stats.skew(data, bias=False),
        'Эксцесс': stats.kurtosis(data, bias=False)
    })
df_desc = pd.DataFrame(desc_stats).set_index('Год')
print(df_desc.round(3))

# Графика: Гистограммы и боксплоты для ПФО
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for idx, y in enumerate(YEARS):
    sns.histplot(df_pfo[y].dropna(), kde=True, ax=axes[idx], color='skyblue')
    axes[idx].set_title(f'Гистограмма ПФО ({y})')
    axes[idx].set_xlabel('Врачей на 10 тыс. населения')
    axes[idx].set_ylabel('Частота')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_pfo[YEARS], palette='Set2')
plt.title('Диаграмма размаха (боксплот) показателя X по годам для ПФО')
plt.xlabel('Год')
plt.ylabel('Число врачей на 10 тыс. населения')
plt.show()

# ---------------------------------------------------------
# ПУНКТ 3: ПРОВЕРКА ГИПОТЕЗЫ О НОРМАЛЬНОСТИ (ПФО)
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("ПУНКТ 3: КРИТЕРИЙ ШАПИРО-УИЛКА ДЛЯ ПФО")
print("=" * 70)

pfo_normality = {}
for y in YEARS:
    data = df_pfo[y].dropna()
    stat, p_val = stats.shapiro(data)
    is_norm = p_val > ALPHA
    pfo_normality[y] = is_norm
    res = "Распределение НОРМАЛЬНОЕ (H0)" if is_norm else "Распределение ОТЛИЧНО от нормального (отвергаем H0)"
    print(f"[{y}] W = {stat:.4f}, p-value = {p_val:.4f} | Решение: {res}")

# ---------------------------------------------------------
# ПУНКТ 4: СРАВНЕНИЕ С ОКРУГОМ-СОСЕДОМ (ПФО vs УФО)
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("ПУНКТ 4: СРАВНЕНИЕ ДВУХ НЕЗАВИСИМЫХ ВЫБОРОК (ПФО vs УФО)")
print("=" * 70)

for y in YEARS:
    s_pfo = df_pfo[y].dropna()
    s_ufo = df_ufo[y].dropna()

    w_pfo, p_pfo = stats.shapiro(s_pfo)
    w_ufo, p_ufo = stats.shapiro(s_ufo)
    norm_both = (p_pfo > ALPHA) and (p_ufo > ALPHA)

    mean_pfo, mean_ufo = s_pfo.mean(), s_ufo.mean()
    alt = 'greater' if mean_pfo > mean_ufo else 'less'
    alt_symbol = ">" if alt == 'greater' else "<"

    if norm_both:
        w_lev, p_lev = stats.levene(s_pfo, s_ufo)
        eq_var = p_lev > ALPHA
        t_stat, p_val = stats.ttest_ind(s_pfo, s_ufo, equal_var=eq_var, alternative=alt)
        test_name = f"t-критерий {'Стьюдента' if eq_var else 'Уэлча'} (равенство дисперсий p={p_lev:.4f})"
        stat_char = "t"
        stat_val = t_stat
    else:
        u_stat, p_val = stats.mannwhitneyu(s_pfo, s_ufo, alternative=alt)
        test_name = "Критерий Манна-Уитни"
        stat_char = "U"
        stat_val = u_stat

    decision = "Принимаем H0 (различия незначимы)" if p_val >= ALPHA else "Отвергаем H0 (различия статистически значимы)"
    print(f"\n[{y} год] H1: ПФО {alt_symbol} УФО | Критерий: {test_name}")
    print(f"   Статистика {stat_char} = {stat_val:.4f}, p-value = {p_val:.4f} -> {decision}")

# ---------------------------------------------------------
# ПУНКТ 5: АНАЛИЗ ДИНАМИКИ (ЗАВИСИМЫЕ ВЫБОРКИ: 2005 vs 2021)
# ---------------------------------------------------------
print("\n" + "=" * 70)
print(f"ПУНКТ 5: ДИНАМИКА В ПФО ({PAIR_YEARS[0]} vs {PAIR_YEARS[1]})")
print("=" * 70)

y1, y2 = PAIR_YEARS
df_paired = df_pfo[['Region', y1, y2]].dropna()
d = df_paired[y2] - df_paired[y1]

w_diff, p_diff = stats.shapiro(d)
norm_diff = p_diff > ALPHA

print(f"Вектор разностей d = ({y2} - {y1}): W = {w_diff:.4f}, p = {p_diff:.4f}")
print(f"Нормальность разностей: {'ДА' if norm_diff else 'НЕТ'}")

if norm_diff:
    t_stat, p_val = stats.ttest_rel(df_paired[y2], df_paired[y1])
    test_used = "Парный t-критерий Стьюдента"
    stat_val = t_stat
else:
    w_stat, p_val = stats.wilcoxon(df_paired[y2], df_paired[y1])
    test_used = "Парный критерий Уилкоксона"
    stat_val = w_stat

dec_pair = "Принимаем H0 (динамика отсутствует)" if p_val >= ALPHA else "Отвергаем H0 (изменения значимы)"
print(f"Критерий: {test_used} | Статистика = {stat_val:.4f}, p-value = {p_val:.4f} -> {dec_pair}")

# ---------------------------------------------------------
# ПУНКТ 6: СРАВНЕНИЕ С ОБЩЕРОССИЙСКИМ УРОВНЕМ
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("ПУНКТ 6: ОДНОВЫБОРОЧНЫЙ АНАЛИЗ (ПФО vs РФ)")
print("=" * 70)

for y in YEARS:
    s_pfo = df_pfo[y].dropna()
    mu_rf = float(rf_values[y])
    mean_pfo = s_pfo.mean()

    alt = 'greater' if mean_pfo > mu_rf else 'less'
    alt_symbol = ">" if alt == 'greater' else "<"

    if pfo_normality[y]:
        t_stat, p_val = stats.ttest_1samp(s_pfo, popmean=mu_rf, alternative=alt)
        test_name = "Одновыборочный t-критерий"
        stat_val = t_stat
    else:
        # Уилкоксон для одной выборки: центрируем относительно mu_rf
        diff = s_pfo - mu_rf
        w_stat, p_val = stats.wilcoxon(diff, alternative=alt)
        test_name = "Критерий знаковых рангов Уилкоксона (1 выборка)"
        stat_val = w_stat

    dec_rf = "Принимаем H0 (уровень ПФО не отличается от РФ)" if p_val >= ALPHA else "Отвергаем H0 (уровень ПФО значимо отличается от РФ)"
    print(f"[{y} год] ПФО ({mean_pfo:.2f}) vs РФ ({mu_rf:.2f}) | H1: ПФО {alt_symbol} РФ")
    print(f"   {test_name}: статистика = {stat_val:.4f}, p-value = {p_val:.4f} -> {dec_rf}")

# ---------------------------------------------------------
# ЧАСТЬ II. ПУНКТ 7: МАКРОРЕГИОНЫ (КРАСКЕЛ-УОЛЛИС: ПФО, УФО, СФО, 2015 г.)
# ---------------------------------------------------------
print("\n" + "=" * 70)
print(f"ПУНКТ 7: СРАВНЕНИЕ ТРЕХ ОКРУГОВ ЗА {KRUSKAL_YEAR} ГОД")
print("=" * 70)

sample_pfo = df_pfo[KRUSKAL_YEAR].dropna()
sample_ufo = df_ufo[KRUSKAL_YEAR].dropna()
sample_sfo = df_sfo[KRUSKAL_YEAR].dropna()

kw_stat, kw_p = stats.kruskal(sample_pfo, sample_ufo, sample_sfo)
dec_kw = "Принимаем H0 (макрорегионы статистически однородны)" if kw_p >= ALPHA else "Отвергаем H0 (между округами есть статистически значимые различия)"

print(f"Критерий Краскела-Уоллиса: H-статистика = {kw_stat:.4f}, p-value = {kw_p:.4f}")
print(f"Решение: {dec_kw}")
print(f"Медианы: ПФО = {sample_pfo.median():.2f}, УФО = {sample_ufo.median():.2f}, СФО = {sample_sfo.median():.2f}")

# Визуализация: совмещенный боксплот
plt.figure(figsize=(8, 6))
plot_df = pd.DataFrame({
    'Число врачей': list(sample_pfo) + list(sample_ufo) + list(sample_sfo),
    'Округ': ['ПФО'] * len(sample_pfo) + ['УФО'] * len(sample_ufo) + ['СФО'] * len(sample_sfo)
})
sns.boxplot(x='Округ', y='Число врачей', data=plot_df, palette='Pastel1')
plt.title(f'Сравнение обеспеченности врачами в ПФО, УФО и СФО ({KRUSKAL_YEAR} г.)')
plt.ylabel('Врачей на 10 тыс. населения')
plt.xlabel('Федеральный округ')
plt.show()