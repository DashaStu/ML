import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ

    # Преобразуем входные данные в numpy массивы для векторизации
    feature_vector = np.array(feature_vector)
    target_vector = np.array(target_vector)

    # 1. Сортируем признаки и переупорядочиваем таргет в соответствии с сортировкой
    sorted_indices = np.argsort(feature_vector)
    feature_sorted = feature_vector[sorted_indices]
    target_sorted = target_vector[sorted_indices]

    # Общее количество элементов
    N = len(target_sorted)

    # Если признак константный или вектор пустой, возвращаем пустые значения
    if N < 2 or feature_sorted[0] == feature_sorted[-1]:
        return np.array([]), np.array([]), None, None

    # 2. Вычисляем возможные пороги (среднее двух соседних)
    # Используем маску для исключения дубликатов признаков (когда соседи равны)
    # feature_sorted[:-1] - все кроме последнего
    # feature_sorted[1:] - все кроме первого (сдвиг на 1)
    distinct_mask = feature_sorted[:-1] != feature_sorted[1:]

    # Пороги вычисляются только там, где значения признаков меняются
    thresholds = (feature_sorted[:-1] + feature_sorted[1:]) / 2.0
    thresholds = thresholds[distinct_mask]

    # Если все значения признака одинаковы (после проверки выше может не сработать для float),
    # distinct_mask будет полностью False
    if len(thresholds) == 0:
        return np.array([]), np.array([]), None, None

    # 3. Векторизованный расчет количества классов 1 и 0 для всех возможных сплитов

    # Кумулятивная сумма единиц (класс 1) слева от сплита
    # cumsum возвращает массив, где i-й элемент сумма от 0 до i.
    # Это соответствует разбиению ПОСЛЕ i-го элемента.
    # Нам нужны все разбиения, кроме самого последнего (весь вектор слева, справа пусто),
    # но distinct_mask уже убирает последний элемент, так как он сравнивает i и i+1.
    # Тем не менее, считаем для всех N-1 позиций.

    cum_target = np.cumsum(target_sorted)

    # Количество объектов в левом поддереве для каждого сплита (от 1 до N-1)
    sizes_left = np.arange(1, N)

    # Количество 1 в левом поддереве
    n1_left = cum_target[:-1]
    # Количество 0 в левом поддереве (Размер левого - кол-во 1)
    n0_left = sizes_left - n1_left

    # Количество объектов в правом поддереве
    sizes_right = N - sizes_left
    # Общее количество 1 во всем векторе
    n1_total = cum_target[-1]

    # Количество 1 в правом поддереве (Общее - Левое)
    n1_right = n1_total - n1_left
    # Количество 0 в правом поддереве
    n0_right = sizes_right - n1_right

    # 4. Расчет критерия Джини H(R) = 1 - p1^2 - p0^2

    # Доли классов (p)
    p1_l = n1_left / sizes_left
    p0_l = n0_left / sizes_left

    p1_r = n1_right / sizes_right
    p0_r = n0_right / sizes_right

    # Джини для левого и правого поддеревьев
    h_left = 1 - p1_l ** 2 - p0_l ** 2
    h_right = 1 - p1_r ** 2 - p0_r ** 2

    # 5. Итоговый функционал Q(R)
    # Q(R) = - (|Rl|/|R|) * H(Rl) - (|Rr|/|R|) * H(Rr)
    ginis_all = - (sizes_left / N) * h_left - (sizes_right / N) * h_right

    # 6. Фильтрация дубликатов (применяем маску, рассчитанную в шаге 2)
    ginis = ginis_all[distinct_mask]

    # 7. Поиск лучшего сплита
    # np.argmax возвращает ПЕРВЫЙ индекс максимального элемента.
    # Так как thresholds отсортированы по возрастанию, при совпадении значений
    # будет выбран сплит с минимальным порогом (требование задачи).
    best_idx = np.argmax(ginis)

    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y != sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(1, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_count / current_click
                sorted_categories = list(map(lambda x: x[1], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(map(lambda x: categories_map[x], sub_X[:, feature]))
            else:
                raise ValueError

            if len(feature_vector) == 3:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "Categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[split], node["right_child"])

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        pass

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
