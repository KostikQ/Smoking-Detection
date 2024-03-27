import numpy as np
import scipy.linalg

"""
Таблица 0.95-квантилей распределения хи-квадрат с N степенями свободы
(содержит значения для N=1, ..., 9). Взята из функции chi2inv в MATLAB/Octave
и используется в качестве порога Mahalanobis gating.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}

class KalmanFilter(object):
    """
    Простой фильтр Калмана для отслеживания ограничивающих рамок в пространстве изображений.

    8-мерное пространство состояний
        x, y, a, h, vx, vy, va, vh

    содержит позицию центра ограничивающей рамки (x, y), соотношение сторон a,
    высоту h и их соответствующие скорости.

    Движение объекта моделируется с помощью модели постоянной скорости. Положение
    ограничивающей рамки (x, y, a, h) принимается как прямое наблюдение состояния (линейная
    модель наблюдения).
    """

    def __init__(self, std_weight_position=1.0/20, std_weight_velocity=1.0/160):
        ndim, dt = 4, 1.0

        # Создание матриц модели фильтра Калмана.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Неопределенность движения и наблюдений выбраны относительно текущей оценки состояния.
        # Эти веса контролируют степень неопределенности в модели.
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

    def initiate(self, measurement):
        """Создание трека на основе несвязанного измерения.

        Параметры
        ----------
        measurement : ndarray
            Координаты ограничивающей рамки (x, y, a, h) с центральной позицией (x, y),
            соотношением сторон a и высотой h.

        Возвращает
        -------
        (ndarray, ndarray)
            Возвращает вектор среднего значения (8-мерный) и матрицу ковариации (8x8-мерный)
            нового трека. Наблюдаемые скорости инициализируются со средним значением 0.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Выполнение шага предсказания фильтра Калмана.

        Параметры
        ----------
        mean : ndarray
            8-мерный вектор среднего значения состояния объекта на предыдущем шаге времени.
        covariance : ndarray
            8x8-мерная матрица ковариации состояния объекта на предыдущем шаге времени.

        Возвращает
        -------
        (ndarray, ndarray)
            Возвращает вектор среднего значения и матрицу ковариации предсказанного состояния.
            Наблюдаемые скорости инициализируются со средним значением 0.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )

        return mean, covariance

    def project(self, mean, covariance):
        """Проецирование распределения состояний в пространство измерений.

        Параметры
        ----------
        mean : ndarray
            Среднее значение состояния (8-мерный массив).
        covariance : ndarray
            Матрица ковариации состояния (8x8-мерная).

        Возвращает
        -------
        (ndarray, ndarray)
            Возвращает проекцию среднего значения и матрицу ковариации заданной оценки состояния.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Выполнение шага коррекции фильтра Калмана.

        Параметры
        ----------
        mean : ndarray
            Вектор среднего значения предсказанного состояния (8-мерный).
        covariance : ndarray
            Матрица ковариации состояния (8x8-мерная).
        measurement : ndarray
            4-мерный вектор измерения (x, y, a, h), где (x, y) - центральная позиция,
            a - соотношение сторон и h - высота ограничивающей рамки.

        Возвращает
        -------
        (ndarray, ndarray)
            Возвращает корректированное распределение состояния на основе измерений.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Вычисление расстояния gating между распределением состояний и измерениями.

        Подходящий порог расстояния можно получить из `chi2inv95`. Если
        `only_position` равно True, распределение хи-квадрат имеет 4 степени
        свободы, иначе 2.

        Параметры
        ----------
        mean : ndarray
            Вектор среднего значения состояния (8-мерный).
        covariance : ndarray
            Ковариационная матрица состояния (8x8-мерная).
        measurements : ndarray
            Матрица измерений размерности Nx4, где каждое измерение имеет
            формат (x, y, a, h), где (x, y) - центральная позиция ограничивающей
            рамки, a - соотношение сторон и h - высота.
        only_position : Optional[bool]
            Если значение True, вычисление расстояния выполняется относительно
            центральной позиции ограничивающей рамки.

        Возвращает
        -------
        ndarray
            Возвращает массив длины N, где i-й элемент содержит квадрат расстояния Махаланобиса
            между (mean, covariance) и `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
