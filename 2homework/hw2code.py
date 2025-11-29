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
    feature_vector = np.array(feature_vector)
    target_vector = np.array(target_vector)

    #Сортируем признаки и переупорядочиваем таргет в соответствии с сортировкой
    sorted_indices = np.argsort(feature_vector)
    feature_sorted = feature_vector[sorted_indices]
    target_sorted = target_vector[sorted_indices]

    # Общее количество элементов
    N = len(target_sorted)

    if N < 2 or feature_sorted[0] == feature_sorted[-1]:
        return np.array([]), np.array([]), None, None

    # Вычисляем возможные пороги (среднее двух соседних)
    # Используем маску для исключения дубликатов признаков (когда соседи равны)
    distinct_mask = feature_sorted[:-1] != feature_sorted[1:]

    # Пороги вычисляются только там, где значения признаков меняются
    thresholds = (feature_sorted[:-1] + feature_sorted[1:]) / 2.0
    thresholds = thresholds[distinct_mask]

    # Если все значения признака одинаковы
    # distinct_mask будет полностью False
    if len(thresholds) == 0:
        return np.array([]), np.array([]), None, None

    # Векторизованный расчет количества классов 1 и 0 для всех возможных сплитов
    cum_target = np.cumsum(target_sorted)

    sizes_left = np.arange(1, N)
    n1_left = cum_target[:-1]
    n0_left = sizes_left - n1_left

    sizes_right = N - sizes_left
    n1_total = cum_target[-1]
    n1_right = n1_total - n1_left
    n0_right = sizes_right - n1_right

    #Расчет критерия Джини H(R) = 1 - p1^2 - p0^2

    p1_l = n1_left / sizes_left
    p0_l = n0_left / sizes_left

    p1_r = n1_right / sizes_right
    p0_r = n0_right / sizes_right

    h_left = 1 - p1_l ** 2 - p0_l ** 2
    h_right = 1 - p1_r ** 2 - p0_r ** 2

    # Q(R) = - (|Rl|/|R|) * H(Rl) - (|Rr|/|R|) * H(Rr)
    ginis_all = - (sizes_left / N) * h_left - (sizes_right / N) * h_right

    ginis = ginis_all[distinct_mask]

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

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        # Ошибка 1: условие остановки (все метки одинаковые)
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        # Ошибка 2: индексация признаков с 0
        for feature in range(0, sub_X.shape[1]):
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
                    # Ошибка 3: правильный расчет доли (Target Encoding)
                    ratio[key] = current_click / current_count
                # Ошибка 4: берем ключ категории (x[0]), а не значение ratio (x[1])
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                # Ошибка 5: map в list для корректной работы np.array в Python 3
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError
            # Ошибка 6: удалена странная проверка len == 3

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini is None:
                continue

            # Проверяем, лучше ли этот джини, чем найденный ранее
            if gini_best is None or gini > gini_best:

                # Создаем временный сплит, чтобы проверить min_samples_leaf
                current_split = feature_vector < threshold

                if self._min_samples_leaf is not None:
                    len_left = np.sum(current_split)
                    len_right = np.sum(np.logical_not(current_split))

                    if len_left < self._min_samples_leaf or len_right < self._min_samples_leaf:
                        continue


                feature_best = feature
                gini_best = gini
                split = current_split

                if feature_type == "real":
                    threshold_best = threshold
                # Ошибка 7: исправление регистра строки "categorical"
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            # Ошибка 8: извлечение самого класса из Counter
            node["class"] = Counter(sub_y).most_common(1)[0][0]
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
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        # Ошибка 9: правильный слайс таргета для правого ребенка
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_index = node["feature_split"]
        feature_type = self._feature_types[feature_index]
        feature_value = x[feature_index]

        if feature_type == "real":
            if feature_value < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if feature_value in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
