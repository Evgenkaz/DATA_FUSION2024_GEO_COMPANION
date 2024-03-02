<h2 align="center" style="color:Black"> Data Fusion 2024 Companion </h2>
<h2 align="center" style="color:Black"><b> Задача 1 «Геоаналитика» </b></h2>

<b>Решение:</b>
1. Получаем матрицу признаков на активности клиентов в локациях
2. Используем кросс валидацию <b>MultilabelStratifiedKFold</b> 
3. Для загрузки решения обучаем и сохраняем 7 моделей <b>XGBClassifier</b>
4. Валидация <b>9.48</b>, Паблик <b>9.27</b>

#### Файлы решения

[./submit](https://github.com/Evgenkaz/DATA_FUSION2024_GEO_COMPANION/tree/main/submit) Для отправки в систему с сохраненными моделями;

[./geo_xgb_solution.ipynb](https://github.com/Evgenkaz/DATA_FUSION2024_GEO_COMPANION/blob/main/geo_xgb_solution.ipynb) Блокнот обучения моделей;


#### Образ опубликован и доступен
<b>"image": "kazenov/vtb_contest:latest"</b> - Кто не смог собрать свой может использовать данный

[./requirements](https://github.com/Evgenkaz/DATA_FUSION2024_GEO_COMPANION/blob/main/requirements.txt)   Библиотеки внутри образа;
